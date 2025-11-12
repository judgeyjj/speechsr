"""
音频质量评估指标

包含:
- LSD (Log-Spectral Distance): 对数频谱距离
- SI-SDR (Scale-Invariant Signal-to-Distortion Ratio): 尺度不变信噪比
- PESQ (Perceptual Evaluation of Speech Quality): 可选
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_lsd(pred_audio, target_audio, sr=44100, n_fft=2048, hop_length=512, eps=1e-8):
    """
    计算Log-Spectral Distance (LSD)
    
    LSD是音频超分辨率的标准评估指标，测量预测音频与目标音频的频谱差异。
    
    论文标准:
    - STFT: n_fft=2048, hop_length=512
    - 单位: dB
    - 越低越好（0表示完全相同）
    
    Args:
        pred_audio: 预测音频 [samples] or [batch, samples]
        target_audio: 目标音频 [samples] or [batch, samples]
        sr: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        eps: 防止log(0)的小常数
    
    Returns:
        lsd: Log-Spectral Distance (dB)
    """
    # 确保相同shape
    if pred_audio.shape != target_audio.shape:
        raise ValueError(f"Shape mismatch: pred {pred_audio.shape} vs target {target_audio.shape}")
    
    def _compute_lsd_impl(pred, target):
        # 处理batch维度
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        # 如果有channel维度，取平均
        if pred.dim() == 3:  # [B, C, T]
            pred = pred.mean(dim=1)
            target = target.mean(dim=1)

        device = pred.device

        # 为避免混合精度导致的半精度运算问题，统一转换为float32但保留原设备
        pred = pred.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)

        # 展平成 [B, T]
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        # 计算STFT
        window = torch.hann_window(n_fft, device=device, dtype=torch.float32)

        pred_stft = torch.stft(
            pred,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )

        target_stft = torch.stft(
            target,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )

        # 计算幅度谱
        pred_mag = torch.abs(pred_stft) + eps
        target_mag = torch.abs(target_stft) + eps

        # 计算对数频谱
        pred_log = torch.log10(pred_mag)
        target_log = torch.log10(target_mag)

        # 计算LSD: sqrt(mean((log_pred - log_target)^2))
        diff_sq = (pred_log - target_log).pow(2)
        lsd = torch.sqrt(diff_sq.mean(dim=(-2, -1)))  # [B]

        # 转换为dB，并对batch取平均
        lsd_db = 20.0 * lsd
        return lsd_db.mean()

    device_type = pred_audio.device.type
    if torch.is_autocast_enabled():
        with torch.autocast(device_type=device_type, enabled=False):
            return _compute_lsd_impl(pred_audio, target_audio).item()
    return _compute_lsd_impl(pred_audio, target_audio).item()


def compute_si_sdr(pred_audio, target_audio, eps=1e-8):
    """
    计算Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    
    SI-SDR是语音增强的标准指标，测量信号与噪声的比率。
    
    Args:
        pred_audio: 预测音频 [samples] or [batch, samples]
        target_audio: 目标音频 [samples] or [batch, samples]
        eps: 防止除零
    
    Returns:
        si_sdr: SI-SDR (dB)，越高越好
    """
    # 确保1D或2D
    if pred_audio.dim() == 1:
        pred_audio = pred_audio.unsqueeze(0)
        target_audio = target_audio.unsqueeze(0)
    
    # 如果有channel维度，取平均
    if pred_audio.dim() == 3:
        pred_audio = pred_audio.mean(dim=1)
        target_audio = target_audio.mean(dim=1)
    
    def _compute_si_sdr_impl(pred, target):
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        if pred.dim() == 3:
            pred = pred.mean(dim=1)
            target = target.mean(dim=1)

        device = pred.device
        pred = pred.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)

        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

        alpha = (pred * target).sum(dim=-1, keepdim=True) / (
            target.pow(2).sum(dim=-1, keepdim=True) + eps
        )

        target_scaled = alpha * target
        noise = pred - target_scaled

        si_sdr = 10 * torch.log10(
            target_scaled.pow(2).sum(dim=-1) / (noise.pow(2).sum(dim=-1) + eps)
        )

        return si_sdr.mean()

    device_type = pred_audio.device.type
    if torch.is_autocast_enabled():
        with torch.autocast(device_type=device_type, enabled=False):
            return _compute_si_sdr_impl(pred_audio, target_audio).item()
    return _compute_si_sdr_impl(pred_audio, target_audio).item()


def compute_mse(pred_audio, target_audio):
    """
    计算均方误差 (MSE)
    
    Args:
        pred_audio: 预测音频
        target_audio: 目标音频
    
    Returns:
        mse: 均方误差
    """
    device_type = pred_audio.device.type

    def _compute_mse_impl(pred, target):
        if pred.dim() == 3:
            pred = pred.mean(dim=1)
            target = target.mean(dim=1)

        pred = pred.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)
        return F.mse_loss(pred, target)

    if torch.is_autocast_enabled():
        with torch.autocast(device_type=device_type, enabled=False):
            return _compute_mse_impl(pred_audio, target_audio).item()
    return _compute_mse_impl(pred_audio, target_audio).item()


def evaluate_audio_quality(pred_audio, target_audio, sr=44100):
    """
    全面评估音频质量
    
    Args:
        pred_audio: 预测音频张量
        target_audio: 目标音频张量
        sr: 采样率
    
    Returns:
        metrics: 包含所有指标的字典
    """
    metrics = {}
    
    try:
        metrics['lsd'] = compute_lsd(pred_audio, target_audio, sr=sr)
    except Exception as e:
        print(f"Warning: LSD computation failed: {e}")
        metrics['lsd'] = float('nan')
    
    try:
        metrics['si_sdr'] = compute_si_sdr(pred_audio, target_audio)
    except Exception as e:
        print(f"Warning: SI-SDR computation failed: {e}")
        metrics['si_sdr'] = float('nan')
    
    try:
        metrics['mse'] = compute_mse(pred_audio, target_audio)
    except Exception as e:
        print(f"Warning: MSE computation failed: {e}")
        metrics['mse'] = float('nan')
    
    return metrics


# 测试代码
if __name__ == "__main__":
    print("=== 测试评估指标 ===")
    
    # 创建测试音频
    sr = 44100
    duration = 2.0  # 2秒
    samples = int(sr * duration)
    
    # 目标音频（正弦波）
    t = torch.linspace(0, duration, samples)
    target = torch.sin(2 * np.pi * 440 * t)  # 440Hz
    
    # 预测音频（略有不同）
    pred = target + 0.05 * torch.randn_like(target)
    
    # 计算指标
    print("\n1. LSD测试:")
    lsd = compute_lsd(pred, target, sr=sr)
    print(f"   LSD: {lsd:.4f} dB")
    print(f"   预期: <5 dB (越低越好)")
    
    print("\n2. SI-SDR测试:")
    si_sdr = compute_si_sdr(pred, target)
    print(f"   SI-SDR: {si_sdr:.2f} dB")
    print(f"   预期: >10 dB (越高越好)")
    
    print("\n3. MSE测试:")
    mse = compute_mse(pred, target)
    print(f"   MSE: {mse:.6f}")
    
    print("\n4. 综合评估:")
    metrics = evaluate_audio_quality(pred, target, sr=sr)
    for k, v in metrics.items():
        print(f"   {k.upper()}: {v:.4f}")
    
    print("\n=== 测试通过 ===")
