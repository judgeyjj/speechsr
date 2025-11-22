import os
import random
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
import scipy.signal as signal


class SAGASRDataset(Dataset):
    """
    SAGA-SR数据集：提供配对的(低分辨率, 高分辨率)音频
    
    论文标准：
    - 使用低通滤波器（4种随机）生成低分辨率音频
    - 截止频率: 2-16kHz 均匀随机
    - 滤波器阶数: 2-10 随机
    - 音频长度: 1.48秒 (44.1kHz下为65536采样点) 这里和论文不一样，在训练代码里面体现了
    """
    
    def __init__(self, audio_dir, sample_rate=44100, duration=5.94,
                 compute_rolloff=True, num_samples=None, audio_channels: int = 1):
        """
        Args:
            audio_dir: 高分辨率音频目录
            sample_rate: 采样率 (论文标准: 44100)
            duration: 音频时长秒数 (论文标准: 5.94)
            compute_rolloff: 是否计算roll-off特征
            num_samples: 固定采样点数（优先级高于duration）
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.duration = duration
        if num_samples is not None:
            self.num_samples = int(num_samples)
        else:
            self.num_samples = int(round(sample_rate * duration))
        self.compute_rolloff = compute_rolloff
        # 目标通道数（与模型配置对齐，例如 Stable Audio 官方权重为 2 通道）
        self.audio_channels = audio_channels
        
        # 获取所有音频文件（递归子目录，兼容大小写扩展名）
        valid_exts = ('.wav', '.mp3', '.flac')
        self.audio_files = []
        for root, _, files in os.walk(audio_dir):
            for fname in files:
                if fname.lower().endswith(valid_exts):
                    self.audio_files.append(os.path.join(root, fname))

        # 确保顺序稳定，便于调试与复现
        self.audio_files.sort()

        if not self.audio_files:
            raise RuntimeError(
                f"在目录 '{audio_dir}' 及其子目录中未找到音频文件，请确认路径或文件扩展名。"
            )
        
        # 滤波器类型（论文标准）
        self.filter_types = ['cheby1', 'butter', 'bessel', 'ellip']
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # 加载高分辨率音频
        audio_path = self.audio_files[idx]
        hr_audio, sr = torchaudio.load(audio_path)
        
        # 重采样到目标采样率
        if sr != self.sample_rate:
            hr_audio = torchaudio.transforms.Resample(sr, self.sample_rate)(hr_audio)
        
        # 单声道处理：先归一成 mono，再根据 audio_channels 再复制
        if hr_audio.shape[0] > 1:
            hr_audio = hr_audio.mean(dim=0, keepdim=True)
        
        # 裁剪或填充到目标长度
        if hr_audio.shape[1] > self.num_samples:
            # 随机裁剪
            start = random.randint(0, hr_audio.shape[1] - self.num_samples)
            hr_audio = hr_audio[:, start:start + self.num_samples]
        elif hr_audio.shape[1] < self.num_samples:
            # 循环填充
            repeats = self.num_samples // hr_audio.shape[1] + 1
            hr_audio = hr_audio.repeat(1, repeats)[:, :self.num_samples]
        
        # 若模型为多通道（如 2 通道），在裁剪/补齐后再复制通道
        if self.audio_channels > 1:
            hr_audio = hr_audio.repeat(self.audio_channels, 1)
        
        # 生成低分辨率音频（论文标准：低通滤波）
        # 注意：低通在单通道上进行，然后按需要复制通道
        lr_audio_np, cutoff_freq = self._apply_lowpass_filter(hr_audio[0].cpu().numpy())
        lr_audio = torch.from_numpy(lr_audio_np).unsqueeze(0).float()
        if self.audio_channels > 1:
            lr_audio = lr_audio.repeat(self.audio_channels, 1)

        # 准备metadata
        metadata = {
            'lr_audio': lr_audio,
            'audio_path': audio_path,
            'lr_cutoff_hz': float(cutoff_freq),
        }
        
        # 读取文本转录（若存在同名 .txt）
        transcript_path = os.path.splitext(audio_path)[0] + ".txt"
        if os.path.isfile(transcript_path):
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    metadata["transcript"] = f.read().strip()
            except Exception as exc:
                print(f"[Dataset] Failed to read transcript for {audio_path}: {exc}")
        
        # 计算roll-off特征（如果需要）
        if self.compute_rolloff:
            from spectral_features import compute_spectral_rolloff
            # compute_spectral_rolloff 内部会对多通道做平均，这里取第一个通道即可
            metadata['rolloff_low'] = compute_spectral_rolloff(
                lr_audio[0], self.sample_rate, rolloff_percent=0.985
            )
            metadata['rolloff_high'] = compute_spectral_rolloff(
                hr_audio[0], self.sample_rate, rolloff_percent=0.985
            )
        
        # 返回 shape 为 [C, T] 的张量（与模型配置的 audio_channels 对齐）
        return hr_audio, metadata
    
    def _apply_lowpass_filter(self, audio):
        """
        应用低通滤波器生成低分辨率音频
        
        论文标准:
        - 4种滤波器: Chebyshev, Butterworth, Bessel, Elliptic
        - 截止频率: 2-16kHz 均匀随机
        - 滤波器阶数: 2-10 随机
        """
        # 随机选择滤波器参数
        filter_type = random.choice(self.filter_types)
        cutoff_freq = random.uniform(2000, 16000)  # Hz
        order = random.randint(2, 10)

        # 设计滤波器
        try:
            if filter_type == 'cheby1':
                # Chebyshev Type I: ripple=0.5dB
                b, a = signal.cheby1(order, 0.5, cutoff_freq, 
                                    btype='low', fs=self.sample_rate)
            elif filter_type == 'butter':
                # Butterworth
                b, a = signal.butter(order, cutoff_freq, 
                                    btype='low', fs=self.sample_rate)
            elif filter_type == 'bessel':
                # Bessel
                b, a = signal.bessel(order, cutoff_freq, 
                                    btype='low', fs=self.sample_rate)
            elif filter_type == 'ellip':
                # Elliptic: ripple=0.5dB, attenuation=40dB
                b, a = signal.ellip(order, 0.5, 40, cutoff_freq, 
                                   btype='low', fs=self.sample_rate)
            
            # 应用滤波器（使用filtfilt保证零相位延迟）
            filtered_audio = signal.filtfilt(b, a, audio)
            
            return filtered_audio.astype(np.float32), cutoff_freq
            
        except Exception as e:
            print(f"Filter design failed ({filter_type}, order={order}, "
                  f"cutoff={cutoff_freq}): {e}")
            # 降级方案：简单低通
            b, a = signal.butter(4, cutoff_freq, btype='low', fs=self.sample_rate)
            return signal.filtfilt(b, a, audio).astype(np.float32), cutoff_freq


# 辅助函数：验证数据集
def validate_dataset(dataset, num_samples=5):
    """验证数据集是否正确生成"""
    print(f"Dataset size: {len(dataset)}")
    
    for i in range(min(num_samples, len(dataset))):
        hr_audio, metadata = dataset[i]
        lr_audio = metadata['lr_audio']
        
        print(f"\nSample {i}:")
        print(f"  HR audio shape: {hr_audio.shape}")
        print(f"  LR audio shape: {lr_audio.shape}")
        print(f"  HR max: {hr_audio.abs().max():.4f}")
        print(f"  LR max: {lr_audio.abs().max():.4f}")
        
        if 'rolloff_low' in metadata:
            print(f"  Rolloff low: {metadata['rolloff_low']:.2f} Hz")
            print(f"  Rolloff high: {metadata['rolloff_high']:.2f} Hz")


if __name__ == "__main__":
    # 测试代码
    dataset = SAGASRDataset(
        audio_dir="/data01/audio_group/m24_yuanjiajun/mdct_sr/data/wav48/eval/p225",
        compute_rolloff=False  # 先不计算rolloff（需要spectral_features.py）
    )
    validate_dataset(dataset, num_samples=3)
