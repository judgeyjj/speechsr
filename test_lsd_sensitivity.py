import math

import torch
import torchaudio
import argparse

from metrics import compute_lsd


def generate_sine(freq=1000.0, sr=44100, duration=1.0, amp=0.5, device="cpu"):
    """生成单频正弦波信号。"""
    num_samples = int(sr * duration)
    t = torch.linspace(0.0, duration, num_samples, endpoint=False, device=device)
    return amp * torch.sin(2 * math.pi * freq * t)


def add_noise_with_snr(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """按给定 SNR(dB) 向信号中加入高斯白噪声。"""
    # 保证是 1D 向量
    if x.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got {x.shape}")

    power = x.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = power / snr_linear
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    return x + noise


def time_shift(x: torch.Tensor, shift_samples: int) -> torch.Tensor:
    """对信号做简单的零填充平移（正数为向前，负数为向后）。"""
    if x.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got {x.shape}")

    if shift_samples == 0:
        return x.clone()

    if abs(shift_samples) >= x.numel():
        # 位移超过长度时，直接返回全零
        return torch.zeros_like(x)

    if shift_samples > 0:
        # 向前平移：前面补零，后面截断
        return torch.cat([torch.zeros(shift_samples, device=x.device), x[:-shift_samples]], dim=0)
    else:
        # 向后平移：后面补零，前面截断
        shift_samples = -shift_samples
        return torch.cat([x[shift_samples:], torch.zeros(shift_samples, device=x.device)], dim=0)


def _compute_lsd_wrapper(a: torch.Tensor, b: torch.Tensor, sr: int, desc: str):
    """使用项目中的 compute_lsd 计算并打印结果。"""
    # 转成 [B, T]，与 metrics.compute_lsd 的 batch 接口对齐
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch in test: {a.shape} vs {b.shape}")

    lsd_value = compute_lsd(a, b, sr=sr)
    print(f"{desc:30s} -> LSD = {lsd_value:.6f} dB")


def main():
    device = "cpu"
    parser = argparse.ArgumentParser(description="LSD 敏感性测试（使用真实语音）")
    parser.add_argument(
        "--wav_path",
        type=str,
        default="/data01/audio_group/m24_yuanjiajun/mdct_sr/data/wav48/eval/p225/p225_001.wav",
        help="用作参考信号的 WAV 文件路径",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=44100,
        help="如果不为 None 且与原始采样率不同，则重采样到该采样率",
    )
    args = parser.parse_args()

    print("=== 加载参考语音 ===")
    ref, sr = torchaudio.load(args.wav_path)
    if ref.dim() == 2 and ref.shape[0] > 1:
        ref = ref.mean(dim=0)
    elif ref.dim() == 2:
        ref = ref[0]
    ref = ref.to(device)

    if args.target_sr is not None and sr != args.target_sr:
        ref = torchaudio.transforms.Resample(sr, args.target_sr)(ref.unsqueeze(0)).squeeze(0)
        sr = args.target_sr

    print("\n=== 基本一致性检查（LSD 应非常接近 0） ===")
    _compute_lsd_wrapper(ref, ref, sr, "同一信号对比自身")

    print("\n=== 增益变化敏感性（整体放大/缩小） ===")
    _compute_lsd_wrapper(ref * 1.05, ref, sr, "增益 +5%")
    _compute_lsd_wrapper(ref * 0.95, ref, sr, "增益 -5%")

    print("\n=== 时间错位敏感性（sample 级平移） ===")
    for shift in [1, 4, 16, 64]:
        _compute_lsd_wrapper(time_shift(ref, shift), ref, sr, f"前移 {shift} 个采样点")
        _compute_lsd_wrapper(time_shift(ref, -shift), ref, sr, f"后移 {shift} 个采样点")

    print("\n=== 噪声敏感性（固定 SNR） ===")
    for snr in [40, 30, 20, 10]:
        noisy = add_noise_with_snr(ref, snr_db=snr)
        _compute_lsd_wrapper(noisy, ref, sr, f"SNR = {snr:2d} dB 噪声")

    print("\n测试完成。可以根据以上不同情况的 LSD 数值，直观感受该指标的敏感性。")


if __name__ == "__main__":
    main()
