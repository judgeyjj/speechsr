import os
import json
import math
import random
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from typing import Dict, Iterable, Optional

try:
    import swanlab  # type: ignore
except ImportError:
    swanlab = None  # SwanLab is optional

from stable_audio_tools.inference.sampling import sample_discrete_euler
from stable_audio_tools.training.utils import InverseLR

from dataset import SAGASRDataset
from conditioner_rolloff import RolloffFourierConditioner, CombinedConditioner
from audio_captioning_adapter import QwenAudioCaptioner, CaptionCache
from metrics import compute_lsd, compute_si_sdr, evaluate_audio_quality


def saga_collate_fn(batch):
    """
    自定义collate函数，正确处理metadata字典
    
    Args:
        batch: list of (hr_audio, metadata) tuples
    
    Returns:
        hr_audio_batch: [B, 1, samples] (VAE需要channel维度)
        metadata_list: list of metadata dicts
    """
    hr_audios = []
    metadatas = []
    
    for hr_audio, metadata in batch:
        # 确保hr_audio有channel维度: [samples] -> [1, samples]
        if hr_audio.dim() == 1:
            hr_audio = hr_audio.unsqueeze(0)
        hr_audios.append(hr_audio)
        metadatas.append(metadata)
    
    # Stack audio: [B, 1, samples]
    hr_audio_batch = torch.stack(hr_audios)
    
    return hr_audio_batch, metadatas


class SAGASRTrainer(pl.LightningModule):
    """
    SAGA-SR训练器
    
    论文标准:
    - 训练目标: Flow Matching
    - 优化器: AdamW (β1=0.9, β2=0.999)
    - 学习率: 1.0e-5
    - 批大小: 256 (可根据GPU调整)
    - 训练步数: 26,000
    """
    
    def __init__(self, model_config_path, learning_rate=1e-5, use_caption=False,
                 swanlab_run: Optional[object] = None, val_save_audio_dir: Optional[str] = None,
                 val_save_interval: int = 1):
        """
        Args:
            model_config_path: Stable Audio模型配置文件路径
            learning_rate: 学习率 (论文标准: 1e-5)
            use_caption: 是否使用文本caption
            val_save_audio_dir: 验证阶段保存模型生成音频的目录（默认不保存）
            val_save_interval: 保存频率（单位：epoch），最小为1
        """
        super().__init__()
        self.save_hyperparameters(ignore=['swanlab_run'])
        
        self.lr = learning_rate
        self.use_caption = use_caption
        self.swanlab_run = swanlab_run
        self.val_save_audio_dir = val_save_audio_dir
        self.val_save_interval = max(1, val_save_interval)
        self._val_preview: Optional[tuple] = None
        self.captioner: Optional[QwenAudioCaptioner] = None
        
        # 加载Stable Audio模型
        from stable_audio_tools.models.factory import create_model_from_config
        
        with open(model_config_path) as f:
            config = json.load(f)
        
        self.model = create_model_from_config(config)
        self.config = config

        self._freeze_non_dit_modules()
        dit_core = self._get_dit()
        if not hasattr(dit_core, "to_prepend_embed"):
            dit_core.to_prepend_embed = nn.Identity()
        
        # 创建Roll-off条件器
        # 注意：embedding_dim_cross必须匹配T5的维度（768）才能拼接
        self.rolloff_conditioner = RolloffFourierConditioner(
            embedding_dim_cross=768,  # 匹配T5-base的维度
            embedding_dim_global=config['model']['diffusion']['config']['global_cond_dim'],
            dropout_rate=0.1
        )

        self._log_parameter_stats()
        
        # 如果使用caption，加载缓存
        self.caption_cache = None
        if self.use_caption:
            self.caption_cache = CaptionCache('caption_cache.pt')
            if len(self.caption_cache) > 0:
                print(f"Caption cache loaded: {len(self.caption_cache)} entries")

        if self.swanlab_run is not None:
            print("SwanLab logging is enabled.")

    def on_validation_epoch_start(self):
        self._val_preview = None

    def _freeze_non_dit_modules(self):
        """冻结Stable Audio中除DiT之外的模块参数，与论文保持一致。"""
        frozen_modules = []

        if hasattr(self.model, 'pretransform') and self.model.pretransform is not None:
            self.model.pretransform.requires_grad_(False)
            frozen_modules.append('pretransform (VAE)')

        if hasattr(self.model, 'conditioner') and self.model.conditioner is not None:
            self.model.conditioner.requires_grad_(False)
            frozen_modules.append('conditioner (T5)')

        if frozen_modules:
            print(f"Frozen modules: {', '.join(frozen_modules)}")
        else:
            print("Warning: no modules were frozen; please verify model structure")

    def _get_trainable_parameters(self):
        """返回需要训练的参数集合（DiT核心 + 自定义条件器）。"""
        trainable_params = []

        if hasattr(self.model, 'model') and self.model.model is not None:
            trainable_params.extend(p for p in self.model.model.parameters() if p.requires_grad)

        if hasattr(self, 'rolloff_conditioner') and self.rolloff_conditioner is not None:
            trainable_params.extend(p for p in self.rolloff_conditioner.parameters() if p.requires_grad)

        return trainable_params

    def _log_parameter_stats(self):
        """打印可训练与冻结参数统计，便于核对论文设定。"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self._get_trainable_parameters())
        non_trainable_params = total_params - trainable_params

        bytes_per_param = 4  # 默认float32
        model_size_mb = total_params * bytes_per_param / (1024 ** 2)
        ratio = trainable_params / total_params if total_params > 0 else 0.0

        print("=== Parameter Statistics ===")
        print(f"Trainable params : {trainable_params / 1e6:.3f} M")
        print(f"Frozen params    : {non_trainable_params / 1e6:.3f} M")
        print(f"Total params     : {total_params / 1e6:.3f} M")
        print(f"Model size (MB)  : {model_size_mb:.3f} MB (float32)")
        print(f"Trainable ratio  : {ratio * 100:.2f}%")
        print("============================")

        self._swanlab_log({
            "model/trainable_params_m": trainable_params / 1e6,
            "model/trainable_ratio": ratio
        }, step=0)

    def _swanlab_log(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.swanlab_run is None or swanlab is None:
            return
        if step is None:
            step = self.global_step
        try:
            swanlab.log(metrics, step=step)
        except Exception as exc:  # 防御性处理，避免日志失败中断训练
            print(f"[SwanLab] Logging failed: {exc}")

    def _compute_grad_norm(self) -> Optional[float]:
        total = 0.0
        found = False
        for param in self._get_trainable_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            total += grad.float().norm(2).square().item()
            found = True
        if not found:
            return None
        return math.sqrt(total)

    def _get_dit(self):
        """返回底层 DiffusionTransformer 模型。"""
        model = self.model.model  # DiTWrapper
        if hasattr(model, "model"):
            return model.model
        raise AttributeError("DiffusionTransformer not found in wrapped model.")

    def _build_rolloff_prepend(self, rolloff_global: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        根据论文要求，将 roll-off 全局嵌入与当前时间步嵌入相加，构造 prepend token。
        Args:
            rolloff_global: [B, D] roll-off 全局嵌入
            timesteps: [B] 扩散时间步 (0~1)
        Returns:
            rolloff_prepend: [B, 1, D]
        """
        dit = self._get_dit()
        timestep_features = dit.timestep_features(timesteps[:, None])
        timestep_embed = dit.to_timestep_embed(timestep_features)  # [B, D]
        rolloff_prepend = rolloff_global.unsqueeze(1) + timestep_embed.unsqueeze(1)
        return rolloff_prepend

    
    def training_step(self, batch, batch_idx):
        """
        训练步骤
        
        论文Flow Matching目标:
        z_t = (1-t)*noise + t*z_h
        v_target = z_h - noise
        loss = ||v_pred - v_target||^2
        """
        hr_audio, metadata = batch
        
        # 提取低分辨率音频 (metadata 现在是字典列表)
        lr_audio = torch.stack([m['lr_audio'] for m in metadata]).to(self.device)
        
        # 编码为latent
        with torch.no_grad():
            lr_latent = self.model.pretransform.encode(lr_audio)  # [B, 64, L]
            hr_latent = self.model.pretransform.encode(hr_audio.to(self.device))  # [B, 64, L]
        
        # Flow Matching (论文公式)
        batch_size = hr_audio.shape[0]
        t = torch.rand(batch_size, device=self.device)  # [B], 均匀分布 [0,1]
        noise = torch.randn_like(hr_latent)  # [B, 64, L]
        
        # 插值
        z_t = (1 - t[:, None, None]) * noise + t[:, None, None] * hr_latent
        
        # 目标速度
        v_target = hr_latent - noise
        
        # 准备metadata（Stable Audio标准格式）
        # 每个metadata字典必须包含所有conditioning keys
        for i, m in enumerate(metadata):
            # 1. 文本条件
            caption = self._resolve_caption(metadata[i])
            # 论文标准: 10% text dropout for CFG training
            if self.use_caption and random.random() < 0.1:
                caption = ""
            m['prompt'] = caption
            
            # 2. 时间条件（Stable Audio必需）
            m['seconds_start'] = 0
            m['seconds_total'] = 1.48
            
            # 3. padding_mask（Stable Audio需要）
            if 'padding_mask' not in m:
                # 创建全1的padding mask（表示全部是有效音频）
                m['padding_mask'] = torch.ones(hr_audio.shape[-1], dtype=torch.bool)
        
        # 使用Stable Audio的conditioner处理metadata
        conditioning = self.model.conditioner(metadata, self.device)
        
        # Roll-off条件处理
        rolloff_low = torch.stack([m['rolloff_low'] for m in metadata])
        rolloff_high = torch.stack([m['rolloff_high'] for m in metadata])
        rolloff_cond = self.rolloff_conditioner(rolloff_low, rolloff_high, apply_dropout=True)
        
        # 模型前向传播
        # 获取conditioning输入
        conditioning_inputs = self.model.get_conditioning_inputs(conditioning)
        
        # 集成roll-off条件（双通道注入）
        if rolloff_cond['cross_attn'] is not None:
            # 通道1: 拼接到Cross-Attention
            if conditioning_inputs.get('cross_attn_cond') is not None:
                conditioning_inputs['cross_attn_cond'] = torch.cat([
                    conditioning_inputs['cross_attn_cond'],  # 文本嵌入 [B, seq_len, D]
                    rolloff_cond['cross_attn']  # Roll-off嵌入 [B, 1, D]
                ], dim=1)
            else:
                conditioning_inputs['cross_attn_cond'] = rolloff_cond['cross_attn']

        if rolloff_cond['global'] is not None:
            rolloff_prepend = self._build_rolloff_prepend(rolloff_cond['global'], t)
            conditioning_inputs['prepend_cond'] = rolloff_prepend
            conditioning_inputs['prepend_cond_mask'] = torch.ones(
                rolloff_prepend.shape[:2], device=self.device, dtype=torch.bool
            )
        
        # 添加SAGA-SR的input_concat_cond（低分辨率latent）
        conditioning_inputs['input_concat_cond'] = lr_latent
        
        # 调用DiT模型
        v_pred = self.model.model(
            z_t, 
            t,
            **conditioning_inputs  # 展开所有conditioning参数（包括input_concat_cond）
        )
        
        # Flow Matching损失
        loss = F.mse_loss(v_pred, v_target)
        # 记录
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/rolloff_low_mean', rolloff_low.mean(), on_step=False, on_epoch=True)
        self.log('train/rolloff_high_mean', rolloff_high.mean(), on_step=False, on_epoch=True)

        self._swanlab_log({
            "train/loss": loss.item(),
            "train/rolloff_low_mean": rolloff_low.mean().item(),
            "train/rolloff_high_mean": rolloff_high.mean().item(),
        })

        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        验证步骤：计算LSD和其他评估指标
        
        论文评估指标:
        - LSD (Log-Spectral Distance): 主要指标，越低越好
        - SI-SDR: 辅助指标，越高越好
        """
        hr_audio, metadata = batch
        
        # 提取低分辨率音频
        lr_audio = torch.stack([m['lr_audio'] for m in metadata]).to(self.device)
        
        # 编码为latent
        with torch.no_grad():
            lr_latent = self.model.pretransform.encode(lr_audio)
            hr_latent = self.model.pretransform.encode(hr_audio.to(self.device))
        
        # 准备metadata
        for i, m in enumerate(metadata):
            m['prompt'] = self._resolve_caption(m, training=False)
            m['seconds_start'] = 0
            m['seconds_total'] = 1.48
            if 'padding_mask' not in m:
                m['padding_mask'] = torch.ones(hr_audio.shape[-1], dtype=torch.bool)
        
        # 使用模型生成高分辨率音频
        conditioning = self.model.conditioner(metadata, self.device)
        
        # Roll-off条件
        rolloff_low = torch.stack([m['rolloff_low'] for m in metadata])
        rolloff_high = torch.stack([m['rolloff_high'] for m in metadata])
        rolloff_cond = self.rolloff_conditioner(rolloff_low, rolloff_high, apply_dropout=False)
        
        conditioning_inputs = self.model.get_conditioning_inputs(conditioning)
        text_cross_attn = conditioning_inputs.get('cross_attn_cond')
        
        if rolloff_cond['cross_attn'] is not None:
            if text_cross_attn is not None:
                conditioning_inputs['cross_attn_cond'] = torch.cat([
                    text_cross_attn,
                    rolloff_cond['cross_attn']
                ], dim=1)
            else:
                conditioning_inputs['cross_attn_cond'] = rolloff_cond['cross_attn']
        
        conditioning_inputs['input_concat_cond'] = lr_latent
        conditioning_inputs_sampling = conditioning_inputs.copy()
        conditioning_inputs_sampling['text_cross_attn_cond'] = text_cross_attn

        pred_latent = self._sample_with_cfg(
            batch_size=hr_audio.shape[0],
            lr_latent=lr_latent,
            rolloff_cond=rolloff_cond,
            conditioning_inputs=conditioning_inputs_sampling,
            num_steps=100,
            guidance_scale_acoustic=1.4,
            guidance_scale_text=1.2,
        )
        
        # 解码为音频
        pred_audio = self.model.pretransform.decode(pred_latent)
        
        # 低频替换（使用每个样本的低分辨率截止频率）
        pred_audio = self._low_frequency_replace(pred_audio, lr_audio, rolloff_low)

        # 仅保存当前epoch的首个样本
        if (
            self.val_save_audio_dir is not None
            and not self.trainer.sanity_checking
            and self._val_preview is None
        ):
            base_name = os.path.splitext(os.path.basename(metadata[0]['audio_path']))[0]
            preview_audio = pred_audio[0].detach().cpu()
            # 保存要求: torchaudio.save 对 WAV 期望 float32/PCM 等，避免混合精度下的 float16
            if preview_audio.dtype != torch.float32:
                preview_audio = preview_audio.float()
            if preview_audio.dim() == 1:
                preview_audio = preview_audio.unsqueeze(0)
            self._val_preview = (
                preview_audio.contiguous(),
                base_name,
            )
        
        # 计算LSD和SI-SDR
        pred_audio_flat = pred_audio.cpu().flatten()
        hr_audio_flat = hr_audio.flatten()
        
        # 确保长度匹配
        min_len = min(len(pred_audio_flat), len(hr_audio_flat))
        pred_audio_flat = pred_audio_flat[:min_len]
        hr_audio_flat = hr_audio_flat[:min_len]
        
        lsd = compute_lsd(pred_audio_flat, hr_audio_flat, sr=44100)
        si_sdr = compute_si_sdr(pred_audio_flat, hr_audio_flat)
        
        # 记录
        self.log('val/lsd', lsd, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/si_sdr', si_sdr, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/rolloff_low_mean', rolloff_low.mean(), on_step=False, on_epoch=True)
        self.log('val/rolloff_high_mean', rolloff_high.mean(), on_step=False, on_epoch=True)

        self._swanlab_log({
            "val/lsd": float(lsd),
            "val/si_sdr": float(si_sdr),
            "val/rolloff_low_mean": rolloff_low.mean().item(),
            "val/rolloff_high_mean": rolloff_high.mean().item(),
        })

        return {'val_lsd': lsd, 'val_si_sdr': si_sdr}

    def on_validation_epoch_end(self):
        if self.val_save_audio_dir is None or self.trainer.sanity_checking:
            self._val_preview = None
            return

        if self.current_epoch % self.val_save_interval != 0:
            self._val_preview = None
            return

        if self._val_preview is None:
            return

        preview_audio, base_name = self._val_preview
        if preview_audio.dim() == 3:
            preview_audio = preview_audio.squeeze(0)
        if preview_audio.dim() == 1:
            preview_audio = preview_audio.unsqueeze(0)
        # 确保为 float32 并限制幅度到 [-1, 1]
        if preview_audio.dtype != torch.float32:
            preview_audio = preview_audio.float()
        preview_audio = preview_audio.clamp_(-1.0, 1.0)

        epoch_dir = os.path.join(self.val_save_audio_dir, f"epoch_{self.current_epoch:04d}")
        os.makedirs(epoch_dir, exist_ok=True)
        output_path = os.path.join(epoch_dir, f"pred_{base_name}.wav")

        sample_rate = self.config.get('sample_rate', 44100)
        torchaudio.save(output_path, preview_audio.contiguous(), sample_rate)
        print(f"[Validation] Saved preview audio: {output_path}")

        self._val_preview = None

    def on_train_end(self):
        if self.use_caption and self.caption_cache is not None and len(self.caption_cache) > 0:
            try:
                self.caption_cache.save()
            except Exception as exc:
                print(f"[Caption] Failed to save cache: {exc}")

    def on_after_backward(self):
        if self.swanlab_run is None:
            return
        grad_norm = self._compute_grad_norm()
        if grad_norm is not None:
            self._swanlab_log({"train/dit_grad_norm": grad_norm})

    def _low_frequency_replace(
        self,
        generated: torch.Tensor,
        lr_audio: torch.Tensor,
        cutoff_hz,
    ) -> torch.Tensor:
        """
        用原始低分辨率音频在其 roll-off 截止频率以下的频段替换生成音频。

        Args:
            generated: 生成的高分辨率音频 [B, C, T] 或 [B, T]
            lr_audio: 原始低分辨率音频 [B, C, T] 或 [B, T]
            cutoff_hz: 截止频率 (Hz)，可以是标量或 [B] 张量

        Returns:
            低频替换后的音频，与 generated 形状一致。
        """
        if generated.dim() == 3:
            gen = generated.squeeze(1)
        else:
            gen = generated

        if lr_audio.dim() == 3:
            lr = lr_audio.squeeze(1)
        elif lr_audio.dim() == 2:
            lr = lr_audio
        else:
            lr = lr_audio.unsqueeze(0)

        n = gen.shape[-1]
        if lr.shape[-1] != n:
            lr = F.interpolate(
                lr.unsqueeze(1),
                size=n,
                mode='linear',
                align_corners=False,
            ).squeeze(1)

        gen_fft = torch.fft.rfft(gen)
        lr_fft = torch.fft.rfft(lr.to(gen.device))

        freqs = torch.fft.rfftfreq(n, d=1.0 / 44100.0).to(gen.device)
        if not isinstance(cutoff_hz, torch.Tensor):
            cutoff = torch.tensor([cutoff_hz], device=gen.device, dtype=freqs.dtype)
        else:
            cutoff = cutoff_hz.to(gen.device, dtype=freqs.dtype)

        if cutoff.dim() == 0:
            cutoff = cutoff.unsqueeze(0)
        cutoff = cutoff.view(-1, 1)

        mask = freqs.unsqueeze(0) <= cutoff
        gen_fft = torch.where(mask, lr_fft, gen_fft)

        merged = torch.fft.irfft(gen_fft, n=n)

        if generated.dim() == 3:
            return merged.unsqueeze(1)
        return merged

    def _resolve_caption(self, metadata_entry: dict, training: bool = True) -> str:
        """
        根据优先级获取caption:
        1. 预生成缓存
        2. 数据集中附带的文本 (transcript)
        3. Qwen音频字幕生成
        4. 退化为空字符串
        """
        if not self.use_caption:
            return metadata_entry.get('transcript', "")

        audio_path = metadata_entry.get('audio_path')

        # 1. 缓存
        if self.caption_cache is not None and audio_path is not None:
            cached = self.caption_cache.get(audio_path)
            if cached:
                return cached

        # 2. 数据集转录
        transcript = metadata_entry.get('transcript')
        if transcript:
            return transcript

        # 3. 在线生成（懒加载）
        try:
            self._ensure_captioner()
            if self.captioner is not None and audio_path is not None:
                caption = self.captioner.generate_caption(
                    audio_path,
                    use_hr_audio=training,
                )
                if self.caption_cache is not None and caption:
                    self.caption_cache.add(audio_path, caption)
                return caption
        except Exception as exc:
            print(f"[Caption] generation failed for {audio_path}: {exc}")

        return ""

    def _ensure_captioner(self):
        if self.captioner is None:
            try:
                self.captioner = QwenAudioCaptioner(mode='local')
            except Exception as exc:
                print(f"[Caption] Unable to initialize captioner: {exc}")
                self.captioner = None

    def _sample_with_cfg(
        self,
        batch_size: int,
        lr_latent: torch.Tensor,
        rolloff_cond: dict,
        conditioning_inputs: dict,
        num_steps: int,
        guidance_scale_acoustic: float,
        guidance_scale_text: float,
    ) -> torch.Tensor:
        """使用SAGA-SR论文的多重CFG + Euler采样生成latent。"""

        # 构建CFG模型包装器
        sampler_wrapper = _SAGASRCFGWrapper(
            base_model=self.model.model,
            lr_latent=lr_latent,
            rolloff_cond=rolloff_cond,
            conditioning_inputs=conditioning_inputs,
            s_a=guidance_scale_acoustic,
            s_t=guidance_scale_text,
            device=self.device,
        )

        noise = torch.randn_like(lr_latent)

        sampled = sample_discrete_euler(
            model=sampler_wrapper,
            x=noise,
            steps=num_steps,
            sigma_max=1.0,
            dist_shift=None,
            disable_tqdm=True,
        )

        return sampled

    def configure_optimizers(self):
        """
        配置优化器
        
        论文标准:
        - AdamW (β1=0.9, β2=0.999)
        - Learning rate: 1.0e-5
        - Scheduler: InverseLR (可选)
        """
        optimizer = torch.optim.AdamW(
            self._get_trainable_parameters(),
            lr=self.lr,
            betas=(0.9, 0.999)
        )

        scheduler = InverseLR(
            optimizer,
            inv_gamma=1_000_000,
            power=0.5,
            warmup=0.99
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


class _SAGASRCFGWrapper:
    """Stable Audio模型的多重CFG包装器，对齐SAGA-SR的三路条件组合。"""

    def __init__(
        self,
        base_model,
        lr_latent: torch.Tensor,
        rolloff_cond: dict,
        conditioning_inputs: dict,
        s_a: float,
        s_t: float,
        device: torch.device,
    ):
        self.base_model = base_model
        self.lr_latent = lr_latent
        self.rolloff_cond = rolloff_cond
        self.s_a = s_a
        self.s_t = s_t
        self.device = device

        # 提前准备文本条件与完整 cross-attn，避免在采样时重复拼接
        self.cross_attn_text = conditioning_inputs.get('text_cross_attn_cond')
        self.cross_attn_full = conditioning_inputs.get('cross_attn_cond')
        # global_cond 已经是 (原始 + roll-off) 的组合，保持一致性
        self.global_cond = conditioning_inputs.get('global_cond')
        self.prepend_mask = conditioning_inputs.get('prepend_cond_mask')

    def __call__(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        rolloff_cross = self.rolloff_cond.get('cross_attn')
        rolloff_global = self.rolloff_cond.get('global')
        rolloff_prepend = None
        prepend_mask = self.prepend_mask

        if rolloff_global is not None:
            dit = self.base_model.model if hasattr(self.base_model, "model") else self.base_model
            timestep_embed = dit.to_timestep_embed(dit.timestep_features(t[:, None]))
            rolloff_prepend = rolloff_global.unsqueeze(1) + timestep_embed.unsqueeze(1)
            if (
                prepend_mask is None
                or prepend_mask.shape[0] != rolloff_prepend.shape[0]
                or prepend_mask.shape[1] != rolloff_prepend.shape[1]
            ):
                prepend_mask = torch.ones(
                    rolloff_prepend.shape[:2],
                    device=rolloff_prepend.device,
                    dtype=torch.bool,
                )

        # 1. 无条件（drop out cross/global，只保留lr_latent）
        v_uncond = self.base_model(
            x,
            t,
            input_concat_cond=self.lr_latent,
            cross_attn_cond=None,
            global_cond=None,
            prepend_cond=None,
            prepend_cond_mask=None,
        )

        # 2. 仅声学条件（global_cond 已包含原始 + roll-off）
        v_acoustic = self.base_model(
            x,
            t,
            input_concat_cond=self.lr_latent,
            cross_attn_cond=rolloff_cross,
            global_cond=self.global_cond,  # 保持与训练一致
            prepend_cond=rolloff_prepend,
            prepend_cond_mask=prepend_mask,
        )

        # 3. 完整条件（文本 + 声学）
        cross_attn_full = self.cross_attn_full
        if cross_attn_full is None:
            if self.cross_attn_text is not None and rolloff_cross is not None:
                cross_attn_full = torch.cat(
                    [self.cross_attn_text, rolloff_cross],
                    dim=1,
                )
            else:
                cross_attn_full = rolloff_cross

        v_full = self.base_model(
            x,
            t,
            input_concat_cond=self.lr_latent,
            cross_attn_cond=cross_attn_full,
            global_cond=self.global_cond,  # 保持与训练一致
            prepend_cond=rolloff_prepend,
            prepend_cond_mask=prepend_mask,
        )

        # 多重CFG合成
        v = v_uncond + self.s_a * (v_acoustic - v_uncond) + self.s_t * (v_full - v_acoustic)

        return v


def main():
    """主训练函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SAGA-SR Training')
    parser.add_argument('--train_dir', type=str, required=True,
                       help='Training audio directory (high-res)')
    parser.add_argument('--val_dir', type=str, default=None,
                       help='Validation audio directory (high-res)')
    parser.add_argument('--model_config', type=str, required=True,
                       help='Model config JSON file path')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (论文标准: 256, 可根据GPU调整)')
    parser.add_argument('--accumulate_grad_batches', type=int, default=16,
                       help='Gradient accumulation steps (e.g., batch=16 + accum=16 = effective_batch=256)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--max_steps', type=int, default=26000,
                       help='Max training steps (论文标准: 26000)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate (论文标准: 1e-5)')
    parser.add_argument('--use_caption', action='store_true',
                       help='Use text captions (默认已开启)')
    parser.add_argument('--disable_caption', action='store_true',
                       help='Disable text captions (override default enablement)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--val_save_audio_dir', type=str, default=None,
                       help='Directory to save one validation preview audio per epoch')
    parser.add_argument('--val_save_interval', type=int, default=1,
                       help='Save validation preview every N epochs (default: 1)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据集
    print("Creating datasets...")
    train_dataset = SAGASRDataset(
        audio_dir=args.train_dir,
        sample_rate=44100,
        duration=1.48,
        compute_rolloff=True,
        num_samples=65536,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=saga_collate_fn  # 使用自定义collate
    )
    
    val_loader = None
    if args.val_dir:
        val_dataset = SAGASRDataset(
            audio_dir=args.val_dir,
            sample_rate=44100,
            duration=1.48,
            compute_rolloff=True,
            num_samples=65536,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=saga_collate_fn  # 使用自定义collate
        )
    
    # 创建模型
    print("Creating model...")
    swanlab_run = None
    if swanlab is not None:
        swanlab_run = swanlab.init(
            project="saga-sr",
            experiment="baseline",  # 可以起个喜欢的名字
        )
    use_caption = True
    if args.disable_caption:
        use_caption = False
    elif args.use_caption:
        use_caption = True
    model = SAGASRTrainer(
        model_config_path=args.model_config,
        learning_rate=args.learning_rate,
        use_caption=use_caption,
        swanlab_run=swanlab_run,
        val_save_audio_dir=args.val_save_audio_dir,
        val_save_interval=args.val_save_interval
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='saga-sr-{step:06d}',
        save_top_k=3,
        monitor='train/loss',
        mode='min',
        every_n_train_steps=1000
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 根据训练集批次数调整验证频率，确保不超过训练step数量
    val_check_interval = None
    if val_loader:
        train_batches = len(train_loader)
        if train_batches > 0:
            val_check_interval = min(1000, train_batches)

    # Trainer
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed',
        accumulate_grad_batches=args.accumulate_grad_batches,  # 梯度累积
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=50,
        val_check_interval=val_check_interval
    )

    # 训练
    print("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.checkpoint
    )
    
    # 保存最终模型
    final_path = os.path.join(args.output_dir, 'saga_sr_final.ckpt')
    trainer.save_checkpoint(final_path)
    print(f"Training completed! Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
