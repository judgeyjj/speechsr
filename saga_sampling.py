"""SAGA-SR 采样工具，复用论文中的多重CFG + Euler流程。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from stable_audio_tools.inference.sampling import sample_discrete_euler


@dataclass
class ConditioningBundle:
    """封装采样所需的条件输入。"""

    lr_latent: torch.Tensor
    rolloff_cond: Dict[str, Optional[torch.Tensor]]
    cross_attn_text: Optional[torch.Tensor]
    cross_attn_full: Optional[torch.Tensor]
    global_cond: Optional[torch.Tensor]
    prepend_mask: Optional[torch.Tensor] = None


class SAGASRCFGWrapper:
    """Stable Audio 模型的多重 CFG 包装器。"""

    def __init__(
        self,
        base_model,
        bundle: ConditioningBundle,
        guidance_scale_acoustic: float,
        guidance_scale_text: float,
    ) -> None:
        self.base_model = base_model
        self.bundle = bundle
        self.guidance_scale_acoustic = guidance_scale_acoustic
        self.guidance_scale_text = guidance_scale_text

    def __call__(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        lr_latent = self.bundle.lr_latent
        rolloff_cross = self.bundle.rolloff_cond.get("cross_attn")
        rolloff_global = self.bundle.rolloff_cond.get("global")
        cross_attn_text = self.bundle.cross_attn_text
        cross_attn_full = self.bundle.cross_attn_full

        rolloff_prepend = None
        prepend_mask = self.bundle.prepend_mask
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

        # 1. 无条件分支
        v_uncond = self.base_model(
            x,
            t,
            input_concat_cond=lr_latent,
            cross_attn_cond=None,
            global_cond=None,
            prepend_cond=None,
            prepend_cond_mask=None,
        )

        # 2. 声学分支（仅 roll-off）
        v_acoustic = self.base_model(
            x,
            t,
            input_concat_cond=lr_latent,
            cross_attn_cond=rolloff_cross,
            global_cond=self.bundle.global_cond,
            prepend_cond=rolloff_prepend,
            prepend_cond_mask=prepend_mask,
        )

        # 3. 完整条件（文本 + roll-off）
        if cross_attn_full is None:
            if cross_attn_text is not None and rolloff_cross is not None:
                cross_attn_full = torch.cat([cross_attn_text, rolloff_cross], dim=1)
            else:
                cross_attn_full = rolloff_cross

        v_full = self.base_model(
            x,
            t,
            input_concat_cond=lr_latent,
            cross_attn_cond=cross_attn_full,
            global_cond=self.bundle.global_cond,
            prepend_cond=rolloff_prepend,
            prepend_cond_mask=prepend_mask,
        )

        # 多重 CFG 合成
        s_a = self.guidance_scale_acoustic
        s_t = self.guidance_scale_text
        v = v_uncond + s_a * (v_acoustic - v_uncond) + s_t * (v_full - v_acoustic)
        return v


def sample_cfg_euler(
    base_model,
    lr_latent: torch.Tensor,
    conditioning_inputs: Dict[str, Optional[torch.Tensor]],
    rolloff_cond: Dict[str, Optional[torch.Tensor]],
    num_steps: int,
    guidance_scale_acoustic: float,
    guidance_scale_text: float,
) -> torch.Tensor:
    """按 SAGA-SR 论文流程执行 Euler 采样。"""

    bundle = ConditioningBundle(
        lr_latent=lr_latent,
        rolloff_cond=rolloff_cond,
        cross_attn_text=conditioning_inputs.get("text_cross_attn_cond"),
        cross_attn_full=conditioning_inputs.get("cross_attn_cond"),
        global_cond=conditioning_inputs.get("global_cond"),
        prepend_mask=conditioning_inputs.get("prepend_cond_mask"),
    )

    cfg_wrapper = SAGASRCFGWrapper(
        base_model=base_model,
        bundle=bundle,
        guidance_scale_acoustic=guidance_scale_acoustic,
        guidance_scale_text=guidance_scale_text,
    )

    noise = torch.randn_like(lr_latent)

    sampled = sample_discrete_euler(
        model=cfg_wrapper,
        x=noise,
        steps=num_steps,
        sigma_max=1.0,
        dist_shift=None,
        disable_tqdm=True,
    )

    return sampled
