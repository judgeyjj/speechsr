import torch
import torch.nn as nn
import random
from spectral_features import FourierEmbedding


class RolloffFourierConditioner(nn.Module):
    """
    Roll-off双通道条件注入器
    
    - 通道1: Fourier嵌入 -> 拼接到Cross-Attention
    - 通道2: Fourier嵌入 -> 与时间步相加 -> Prepend到DiT输入
    
    继承方式: 独立实现，不修改Stable Audio源码
    """
    
    def __init__(self, 
                 embedding_dim_cross=768,  # Cross-Attention嵌入维度
                 embedding_dim_global=1536,  # Global嵌入维度（匹配DiT）
                 dropout_rate=0.1,
                 min_freq=0.0,
                 max_freq=22050.0):
        """
        Args:
            embedding_dim_cross: Cross-Attention通道嵌入维度
            embedding_dim_global: Global通道嵌入维度（需匹配DiT的global_cond_dim）
            dropout_rate: Dropout概率（保留接口，当前实现不对roll-off做dropout）
            min_freq: Roll-off归一化最小值
            max_freq: Roll-off归一化最大值
        """
        super().__init__()
        
        self.embedding_dim_cross = embedding_dim_cross
        self.embedding_dim_global = embedding_dim_global
        self.dropout_rate = dropout_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        # Fourier嵌入层（共享）
        self.fourier_embed = FourierEmbedding(embed_dim=embedding_dim_cross)
        
        # 通道1: Cross-Attention投影层
        # 输入: [low_emb, high_emb] 拼接 -> embedding_dim_cross * 2
        self.cross_attn_proj = nn.Linear(
            embedding_dim_cross * 2,
            embedding_dim_cross
        )
        
        # 通道2: Global投影层
        # 输入: [low_emb, high_emb] 拼接 -> embedding_dim_global
        self.global_proj = nn.Linear(
            embedding_dim_cross * 2,
            embedding_dim_global
        )
        
        # Layer Normalization
        self.cross_attn_norm = nn.LayerNorm(embedding_dim_cross)
        self.global_norm = nn.LayerNorm(embedding_dim_global)
    
    def _normalize_rolloff(self, rolloff_freq):
        """归一化roll-off频率到 [0, 1)"""
        normalized = (rolloff_freq - self.min_freq) / (self.max_freq - self.min_freq)
        return torch.clamp(normalized, 0.0, 0.999)
    
    def forward(self, rolloff_low, rolloff_high, apply_dropout=True):
        """
        前向传播
        
        Args:
            rolloff_low: 低分辨率音频的roll-off频率 [B] or [B, 1] (Hz)
            rolloff_high: 高分辨率音频的roll-off频率 [B] or [B, 1] (Hz)
            apply_dropout: 是否应用dropout（训练时True，推理时False）
        
        Returns:
            dict:
                'cross_attn': Cross-Attention嵌入 [B, 1, D_cross] or None
                'global': Global嵌入 [B, D_global] or None
        """
        # 目前不对 roll-off 条件做 dropout，始终提供 fh, fl
        # 确保输入是张量
        if not isinstance(rolloff_low, torch.Tensor):
            rolloff_low = torch.tensor([rolloff_low], dtype=torch.float32)
        if not isinstance(rolloff_high, torch.Tensor):
            rolloff_high = torch.tensor([rolloff_high], dtype=torch.float32)
        
        # 移到同一设备
        device = next(self.parameters()).device
        rolloff_low = rolloff_low.to(device)
        rolloff_high = rolloff_high.to(device)
        
        # 归一化
        normalized_low = self._normalize_rolloff(rolloff_low)
        normalized_high = self._normalize_rolloff(rolloff_high)
        
        # Fourier嵌入
        emb_low = self.fourier_embed(normalized_low)  # [B, D_cross]
        emb_high = self.fourier_embed(normalized_high)  # [B, D_cross]
        
        # 拼接低频和高频嵌入
        combined_emb = torch.cat([emb_low, emb_high], dim=-1)  # [B, D_cross*2]
        
        # 通道1: Cross-Attention路径
        cross_attn_emb = self.cross_attn_proj(combined_emb)  # [B, D_cross]
        cross_attn_emb = self.cross_attn_norm(cross_attn_emb)
        cross_attn_emb = cross_attn_emb.unsqueeze(1)  # [B, 1, D_cross]
        
        # 通道2: Global路径
        global_emb = self.global_proj(combined_emb)  # [B, D_global]
        global_emb = self.global_norm(global_emb)
        
        return {
            'cross_attn': cross_attn_emb,
            'global': global_emb
        }
    
    def forward_batch(self, batch_metadata, apply_dropout=True):
        """
        批处理前向传播（用于训练）
        
        Args:
            batch_metadata: 批次元数据字典，包含:
                - 'rolloff_low': [B] 低分辨率roll-off
                - 'rolloff_high': [B] 高分辨率roll-off
            apply_dropout: 是否应用dropout
        
        Returns:
            dict: 与forward相同
        """
        rolloff_low = batch_metadata['rolloff_low']
        rolloff_high = batch_metadata['rolloff_high']
        
        return self.forward(rolloff_low, rolloff_high, apply_dropout)


class CombinedConditioner(nn.Module):
    """
    组合条件器：整合T5文本条件和Roll-off条件
    
    用于在训练/推理时统一处理多种条件
    """
    
    def __init__(self, rolloff_conditioner, t5_conditioner=None):
        """
        Args:
            rolloff_conditioner: RolloffFourierConditioner实例
            t5_conditioner: Stable Audio的T5 conditioner（可选）
        """
        super().__init__()
        self.rolloff_conditioner = rolloff_conditioner
        self.t5_conditioner = t5_conditioner
    
    def forward(self, batch_metadata, apply_dropout=True):
        """
        整合所有条件
        
        Args:
            batch_metadata: 包含所有条件的字典:
                - 'rolloff_low': 低分辨率roll-off
                - 'rolloff_high': 高分辨率roll-off
                - 'prompt': 文本提示（可选）
            apply_dropout: 是否应用dropout
        
        Returns:
            dict:
                'cross_attn': 整合后的Cross-Attention条件 [B, seq_len, D]
                'global': 整合后的Global条件 [B, D]
        """
        # 获取roll-off条件
        rolloff_cond = self.rolloff_conditioner.forward_batch(
            batch_metadata, apply_dropout
        )
        
        # 如果有T5文本条件
        if self.t5_conditioner is not None and 'prompt' in batch_metadata:
            text_cond = self.t5_conditioner(batch_metadata['prompt'])
            
            # 合并Cross-Attention条件（序列拼接）
            if rolloff_cond['cross_attn'] is not None and text_cond is not None:
                cross_attn = torch.cat([
                    text_cond,  # [B, seq_len_text, D]
                    rolloff_cond['cross_attn']  # [B, 1, D]
                ], dim=1)
            else:
                cross_attn = text_cond if text_cond is not None else rolloff_cond['cross_attn']
        else:
            cross_attn = rolloff_cond['cross_attn']
        
        return {
            'cross_attn': cross_attn,
            'global': rolloff_cond['global']
        }


# 测试代码
if __name__ == "__main__":
    print("=== 测试 RolloffFourierConditioner ===")
    
    # 创建条件器
    conditioner = RolloffFourierConditioner(
        embedding_dim_cross=256,
        embedding_dim_global=1536,
        dropout_rate=0.1
    )
    
    # 测试数据
    batch_size = 4
    rolloff_low = torch.tensor([3500, 4000, 5000, 6000], dtype=torch.float32)
    rolloff_high = torch.tensor([15000, 16000, 17000, 18000], dtype=torch.float32)
    
    # 前向传播（训练模式）
    conditioner.train()
    output_train = conditioner(rolloff_low, rolloff_high, apply_dropout=True)
    
    print(f"训练模式输出:")
    if output_train['cross_attn'] is not None:
        print(f"  Cross-Attention shape: {output_train['cross_attn'].shape}")
    else:
        print(f"  Cross-Attention: None (dropout)")
    
    if output_train['global'] is not None:
        print(f"  Global shape: {output_train['global'].shape}")
    else:
        print(f"  Global: None (dropout)")
    
    # 前向传播（推理模式）
    conditioner.eval()
    output_eval = conditioner(rolloff_low, rolloff_high, apply_dropout=False)
    
    print(f"\n推理模式输出:")
    print(f"  Cross-Attention shape: {output_eval['cross_attn'].shape}")
    print(f"  Global shape: {output_eval['global'].shape}")
    
    # 测试批处理接口
    batch_metadata = {
        'rolloff_low': rolloff_low,
        'rolloff_high': rolloff_high
    }
    output_batch = conditioner.forward_batch(batch_metadata, apply_dropout=False)
    print(f"\n批处理输出:")
    print(f"  Cross-Attention shape: {output_batch['cross_attn'].shape}")
    print(f"  Global shape: {output_batch['global'].shape}")
    
    print("\n✅ RolloffFourierConditioner 测试通过！")
