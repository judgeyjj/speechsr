import os
import json
import torch
import torchaudio
import argparse
from typing import Optional

from conditioner_rolloff import RolloffFourierConditioner
from spectral_features import compute_spectral_rolloff
from audio_captioning_adapter import QwenAudioCaptioner
from saga_sampling import sample_cfg_euler


class SAGASRInference:
    """
    SAGA-SR推理器
    
    论文标准:
    - 采样器: Euler sampler
    - 采样步数: 100 steps
    - 多重CFG: v_final = v_uncond + s_a*(v_acoustic - v_uncond) + s_t*(v_text - v_uncond)
    - 引导强度: s_a=1.4 (声学), s_t=1.2 (文本)
    """
    
    def __init__(self, 
                 model_checkpoint_path,
                 model_config_path,
                 device='cuda',
                 use_caption=False):
        """
        Args:
            model_checkpoint_path: 训练好的模型权重路径
            model_config_path: 模型配置文件路径
            device: 设备 ('cuda' or 'cpu')
            use_caption: 是否使用文本caption
        """
        self.device = device
        self.use_caption = use_caption
        
        # 加载模型配置
        with open(model_config_path) as f:
            self.config = json.load(f)
        
        # 创建Stable Audio模型
        from stable_audio_tools.models.factory import create_model_from_config
        self.model = create_model_from_config(self.config)
        dit_core = self.model.model.model if hasattr(self.model.model, "model") else self.model.model
        if not hasattr(dit_core, "to_prepend_embed"):
            dit_core.to_prepend_embed = torch.nn.Identity()
        
        # 加载权重
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # 创建Roll-off条件器
        self.rolloff_conditioner = RolloffFourierConditioner(
            embedding_dim_cross=768,
            embedding_dim_global=self.config['model']['diffusion']['config']['global_cond_dim'],
            dropout_rate=0.1
        ).to(device)
        self.rolloff_conditioner.eval()
        
        # 如果使用caption，创建captioner
        self.captioner = None
        if use_caption:
            try:
                self.captioner = QwenAudioCaptioner(mode='local')
                print("Audio captioner loaded")
            except Exception as e:
                print(f"Warning: Failed to load captioner: {e}")
                print("Proceeding without text captions")
                self.use_caption = False
        
        print(f"Model loaded on {device}")
    
    @torch.no_grad()
    def upsample(self,
                 input_audio_path,
                 output_audio_path: Optional[str] = None,
                 target_rolloff=16000.0,
                 num_steps=100,
                 guidance_scale_acoustic=1.4,
                 guidance_scale_text=1.2,
                 caption: Optional[str] = None):
        """
        音频超分辨率推理
        
        Args:
            input_audio_path: 输入低分辨率音频路径
            output_audio_path: 输出高分辨率音频路径
            target_rolloff: 目标roll-off频率 (Hz)
            num_steps: 采样步数 (论文标准: 100)
            guidance_scale_acoustic: 声学引导强度 (论文标准: 1.4)
            guidance_scale_text: 文本引导强度 (论文标准: 1.2)
            caption: 文本描述（可选，如果不提供则自动生成）
        
        Returns:
            hr_audio: 高分辨率音频张量
        """
        # 加载低分辨率音频
        lr_audio, sr = torchaudio.load(input_audio_path)
        
        # 重采样到44.1kHz
        if sr != 44100:
            lr_audio = torchaudio.transforms.Resample(sr, 44100)(lr_audio)
        
        # 依据模型配置的通道数进行上/下混合（官方权重通常为 2 通道）
        target_channels = int(self.config.get("audio_channels", 1))
        if lr_audio.shape[0] > 1:
            # 先统一转为 mono，再按需要复制到目标通道数
            lr_mono = lr_audio.mean(dim=0, keepdim=True)
        else:
            lr_mono = lr_audio

        if target_channels > 1:
            lr_audio = lr_mono.repeat(target_channels, 1)
        else:
            lr_audio = lr_mono
        
        lr_audio = lr_audio.to(self.device)
        
        # 编码为latent
        lr_latent = self.model.pretransform.encode(lr_audio)  # [1, 64, L]
        
        # 计算roll-off特征
        rolloff_low = compute_spectral_rolloff(lr_audio[0], 44100, rolloff_percent=0.985)
        rolloff_high = target_rolloff
        
        print(f"Input roll-off: {rolloff_low:.2f} Hz")
        print(f"Target roll-off: {rolloff_high:.2f} Hz")
        
        # 生成caption（如果需要）
        if self.use_caption and self.captioner is not None:
            if caption is None:
                print("Generating caption from low-resolution audio...")
                caption = self.captioner.generate_caption(
                    input_audio_path, 
                    use_hr_audio=False  # 推理阶段使用低分辨率
                )
                print(f"Caption: {caption}")
        
        conditioning_inputs = self._prepare_conditioning_inputs(
            lr_latent=lr_latent,
            lr_audio=lr_audio,
            rolloff_low=rolloff_low,
            rolloff_high=rolloff_high,
            caption=caption,
            apply_dropout=False,
        )

        hr_latent = sample_cfg_euler(
            base_model=self.model.model,
            lr_latent=lr_latent,
            conditioning_inputs=conditioning_inputs,
            rolloff_cond=conditioning_inputs['rolloff_cond'],
            num_steps=num_steps,
            guidance_scale_acoustic=guidance_scale_acoustic,
            guidance_scale_text=guidance_scale_text,
        )
        
        # 解码
        hr_audio = self.model.pretransform.decode(hr_latent)
        hr_audio = self._low_frequency_replace(hr_audio, lr_audio, rolloff_low)
        
        # 保存（可选）
        if output_audio_path is not None:
            torchaudio.save(output_audio_path, hr_audio.cpu(), 44100)
            print(f"Saved to: {output_audio_path}")
        
        return hr_audio
    
    def _prepare_conditioning_inputs(
        self,
        lr_latent: torch.Tensor,
        lr_audio: torch.Tensor,
        rolloff_low: torch.Tensor,
        rolloff_high: torch.Tensor,
        caption: str,
        apply_dropout: bool = False,
    ):
        rolloff_low_tensor = torch.tensor([rolloff_low], dtype=torch.float32, device=self.device)
        rolloff_high_tensor = torch.tensor([rolloff_high], dtype=torch.float32, device=self.device)

        rolloff_cond = self.rolloff_conditioner(
            rolloff_low_tensor,
            rolloff_high_tensor,
            apply_dropout=apply_dropout,
        )

        seconds_total = lr_audio.shape[-1] / 44100.0
        metadata = [{
            'prompt': caption if caption is not None else "",
            'seconds_start': 0.0,
            'seconds_total': float(seconds_total),
            'padding_mask': torch.ones(lr_audio.shape[-1], dtype=torch.bool),
        }]

        conditioning = self.model.conditioner(metadata, self.device)
        conditioning_inputs = self.model.get_conditioning_inputs(conditioning)
        text_cross_attn = conditioning_inputs.get("cross_attn_cond")

        if rolloff_cond['cross_attn'] is not None:
            if text_cross_attn is not None:
                conditioning_inputs['cross_attn_cond'] = torch.cat(
                    [text_cross_attn, rolloff_cond['cross_attn']],
                    dim=1
                )
            else:
                conditioning_inputs['cross_attn_cond'] = rolloff_cond['cross_attn']

        conditioning_inputs['text_cross_attn_cond'] = text_cross_attn
        conditioning_inputs['rolloff_cond'] = rolloff_cond

        return conditioning_inputs

    def _low_frequency_replace(
        self,
        generated: torch.Tensor,
        lr_audio: torch.Tensor,
        cutoff_hz,
    ) -> torch.Tensor:
        """
        低频替换：用低分辨率音频在其 roll-off 截止频率以下的频段替换生成音频。
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
            lr = torch.nn.functional.interpolate(
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


def main():
    parser = argparse.ArgumentParser(description='SAGA-SR Inference')
    parser.add_argument('--input', type=str, required=True,
                       help='Input low-resolution audio path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output high-resolution audio path')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Model checkpoint path')
    parser.add_argument('--config', type=str, required=True,
                       help='Model config JSON path')
    parser.add_argument('--target_rolloff', type=float, default=16000.0,
                       help='Target roll-off frequency (Hz)')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='Sampling steps (论文标准: 100)')
    parser.add_argument('--guidance_acoustic', type=float, default=1.4,
                       help='Acoustic guidance scale (论文标准: 1.4)')
    parser.add_argument('--guidance_text', type=float, default=1.2,
                       help='Text guidance scale (论文标准: 1.2)')
    parser.add_argument('--use_caption', action='store_true',
                       help='Use text captions')
    parser.add_argument('--caption', type=str, default=None,
                       help='Manual caption (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 创建推理器
    print("Loading model...")
    inferencer = SAGASRInference(
        model_checkpoint_path=args.checkpoint,
        model_config_path=args.config,
        device=args.device,
        use_caption=args.use_caption
    )
    
    # 执行超分辨率
    print(f"Upsampling: {args.input} -> {args.output}")
    inferencer.upsample(
        input_audio_path=args.input,
        output_audio_path=args.output,
        target_rolloff=args.target_rolloff,
        num_steps=args.num_steps,
        guidance_scale_acoustic=args.guidance_acoustic,
        guidance_scale_text=args.guidance_text,
        caption=args.caption
    )
    
    print("✅ Inference completed!")


if __name__ == "__main__":
    main()
