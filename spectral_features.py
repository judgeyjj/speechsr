import torch
import torch.nn as nn
import numpy as np


def compute_spectral_rolloff(audio, sr, rolloff_percent=0.985):
    """
    计算音频的频谱滚降点
    
    论文标准:
    - STFT参数: n_fft=2048, hop_length=512, window='hann'
    - Roll-off percentage: 0.985
    - 时间轴聚合: 在时间维度求和后再计算roll-off (而非逐帧)
    
    Args:
        audio: 音频张量 [samples] 或 [channels, samples]
        sr: 采样率
        rolloff_percent: 滚降百分比 (默认0.985)
    
    Returns:
        rolloff_freq: 滚降频率 (Hz)
    """
    # 确保是1D
    if audio.dim() == 2:
        audio = audio.mean(0) if audio.shape[0] > 1 else audio[0]
    
    # STFT (论文标准参数)
    stft = torch.stft(
        audio, 
        n_fft=2048, 
        hop_length=512,
        window=torch.hann_window(2048, device=audio.device),
        return_complex=True
    )
    
    # 计算功率谱
    power = torch.abs(stft) ** 2  # [freq_bins, time_frames]
    
    # 论文方法: 在时间轴上求和（聚合）
    power_aggregated = power.sum(dim=1)  # [freq_bins]
    
    # 计算累积和
    cumsum = torch.cumsum(power_aggregated, dim=0)
    
    # 找到滚降点
    threshold = rolloff_percent * cumsum[-1]
    rolloff_bin = torch.searchsorted(cumsum, threshold)
    
    # 转换为频率 (Hz)
    freq_bins = torch.linspace(0, sr / 2, power_aggregated.shape[0], device=audio.device)
    rolloff_freq = freq_bins[rolloff_bin]  # 返回tensor，不调用.item()
    
    return rolloff_freq


def normalize_rolloff(rolloff_freq, min_freq=0.0, max_freq=22050.0):
    """
    将roll-off频率归一化到 [0, 1) 区间
    
    Args:
        rolloff_freq: 原始频率 (Hz)
        min_freq: 最小频率 (默认0)
        max_freq: 最大频率 (默认22050, Nyquist频率 for 44.1kHz)
    
    Returns:
        normalized: 归一化后的值 [0, 1)
    """
    normalized = (rolloff_freq - min_freq) / (max_freq - min_freq)
    return np.clip(normalized, 0.0, 0.999)


class FourierEmbedding(nn.Module):
    """
    Fourier位置嵌入层
    
    论文要求: 将标量roll-off值投影为高维Fourier嵌入
    类似Transformer的位置编码，使用sin/cos周期函数
    """
    
    def __init__(self, embed_dim=256, max_freq=10000.0):
        """
        Args:
            embed_dim: 嵌入维度 (必须是偶数)
            max_freq: 最大频率 (用于生成频率带)
        """
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim必须是偶数"
        
        self.embed_dim = embed_dim
        
        # 创建频率带 (类似Transformer位置编码)
        freq_bands = torch.exp(
            torch.linspace(0, np.log(max_freq), embed_dim // 2)
        )
        self.register_buffer('freq_bands', freq_bands)
    
    def forward(self, x):
        """
        Args:
            x: 输入标量值 [B] 或 [B, 1], 已归一化到 [0, 1)
        
        Returns:
            embeddings: Fourier嵌入 [B, embed_dim]
        """
        # 确保是 [B, 1] 形状
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        # 计算角度
        angles = x * self.freq_bands * 2 * np.pi  # [B, embed_dim//2]
        
        # sin和cos嵌入
        sin_emb = torch.sin(angles)
        cos_emb = torch.cos(angles)
        
        # 拼接
        embeddings = torch.cat([sin_emb, cos_emb], dim=-1)  # [B, embed_dim]
        
        return embeddings


class SpectralRolloffProcessor:
    """
    完整的Spectral Roll-off处理器
    
    集成: 计算 -> 归一化 -> Fourier嵌入
    """
    
    def __init__(self, embed_dim=256, min_freq=0.0, max_freq=22050.0):
        """
        Args:
            embed_dim: Fourier嵌入维度
            min_freq: 归一化最小值
            max_freq: 归一化最大值
        """
        self.embed_dim = embed_dim
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.fourier_embed = FourierEmbedding(embed_dim)
    
    def process(self, audio, sr, rolloff_percent=0.985):
        """
        完整处理流程
        
        Args:
            audio: 音频张量
            sr: 采样率
            rolloff_percent: 滚降百分比
        
        Returns:
            rolloff_freq: 原始频率 (Hz)
            normalized: 归一化值
            embedding: Fourier嵌入
        """
        # 1. 计算roll-off
        rolloff_freq = compute_spectral_rolloff(audio, sr, rolloff_percent)
        
        # 2. 归一化
        normalized = normalize_rolloff(rolloff_freq, self.min_freq, self.max_freq)
        
        # 3. Fourier嵌入
        normalized_tensor = torch.tensor([normalized], dtype=torch.float32)
        if audio.is_cuda:
            normalized_tensor = normalized_tensor.cuda()
        
        embedding = self.fourier_embed(normalized_tensor)
        
        return rolloff_freq, normalized, embedding


# 测试代码
if __name__ == "__main__":
    print("=== 测试 Spectral Roll-off 计算 ===")
    
    # 生成测试音频: 低频 + 高频
    sr = 44100
    duration = 1.0
    t = torch.linspace(0, duration, int(sr * duration))
    
    # 低频信号 (500Hz)
    low_freq_audio = torch.sin(2 * np.pi * 500 * t)
    rolloff_low = compute_spectral_rolloff(low_freq_audio, sr)
    print(f"低频信号 (500Hz) Roll-off: {rolloff_low:.2f} Hz")
    
    # 高频信号 (5000Hz)
    high_freq_audio = torch.sin(2 * np.pi * 5000 * t)
    rolloff_high = compute_spectral_rolloff(high_freq_audio, sr)
    print(f"高频信号 (5000Hz) Roll-off: {rolloff_high:.2f} Hz")
    
    print("\n=== 测试 Fourier Embedding ===")
    
    # 测试Fourier嵌入
    fourier_embed = FourierEmbedding(embed_dim=256)
    
    # 归一化roll-off值
    norm_low = normalize_rolloff(rolloff_low)
    norm_high = normalize_rolloff(rolloff_high)
    print(f"归一化低频: {norm_low:.4f}")
    print(f"归一化高频: {norm_high:.4f}")
    
    # 生成嵌入
    norm_tensor = torch.tensor([[norm_low], [norm_high]], dtype=torch.float32)
    embeddings = fourier_embed(norm_tensor)
    print(f"嵌入形状: {embeddings.shape}")
    print(f"嵌入范围: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    
    print("\n=== 测试完整处理器 ===")
    
    processor = SpectralRolloffProcessor(embed_dim=256)
    rolloff_freq, normalized, embedding = processor.process(low_freq_audio, sr)
    print(f"Roll-off频率: {rolloff_freq:.2f} Hz")
    print(f"归一化值: {normalized:.4f}")
    print(f"嵌入形状: {embedding.shape}")
    
    print("\n✅ 所有测试通过！")
