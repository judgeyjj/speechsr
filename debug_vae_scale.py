import os
import json
import argparse

import torch

from dataset import SAGASRDataset
from stable_audio_tools.models.factory import create_model_from_config


def _to_device(module, device: torch.device):
    if module is None:
        return
    module.to(device)
    for p in module.parameters():
        p.requires_grad_(False)


def compute_vae_scale(model_config_path: str, audio_dir: str, num_examples: int = 8, duration: float = 1.48):
    """测量 VAE latent 的真实标准差，并对比不同 latent 尺度下解码后的音频幅度。

    - 读取 Stable Audio / SAGA-SR 的 model_config
    - 加载 pretransform（AutoencoderPretransform）
    - 从 audio_dir 取若干条 HR 音频，通过 VAE 编码得到 raw latent
    - 估计 raw latent 的 std，以及 encode() 输出的 latent/scale 的 std
    - 解码：
      - encode() 得到的真实 latent（近似重构）
      - N(0,1) 标准高斯 latent
      - N(0, latent_std) 的 latent
    - 打印各自的 RMS / max，用于判断是否存在“生成音频幅度明显放大”的问题。
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 加载 config 与模型
    with open(model_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = create_model_from_config(config)
    _to_device(model, device)

    if not hasattr(model, "pretransform") or model.pretransform is None:
        raise RuntimeError("当前 config 中未找到 pretransform，请确认使用的是 Stable Audio 风格的 model_config。")

    pre = model.pretransform
    _to_device(pre, device)

    scale = getattr(pre, "scale", 1.0)
    print("=== Pretransform 信息 ===")
    print(f"type(pretransform) = {type(pre)}")
    print(f"pretransform.scale (当前配置) = {scale}")

    sample_rate = config.get("sample_rate", 44100)
    sample_size = config.get("sample_size", 65536)
    audio_channels = config.get("audio_channels", 2)

    # 2) 构造一个简单的数据集，只需要 HR 音频即可
    dataset = SAGASRDataset(
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        duration=duration,
        compute_rolloff=False,
        num_samples=sample_size,
        audio_channels=audio_channels,
    )

    num_examples = min(num_examples, len(dataset))
    print(f"从 {audio_dir} 取前 {num_examples} 条样本用于估计 latent 统计量。")

    hr_list = []
    for i in range(num_examples):
        hr_audio, _ = dataset[i]  # [C, T]
        if hr_audio.dim() == 1:
            hr_audio = hr_audio.unsqueeze(0)
        hr_list.append(hr_audio)

    hr_batch = torch.stack(hr_list, dim=0).to(device)  # [B, C, T]

    def audio_stats(x: torch.Tensor, name: str):
        flat = x.view(-1)
        rms = flat.pow(2).mean().sqrt().item()
        mx = flat.abs().max().item()
        print(f"{name:24s}: rms={rms:.6f}, max={mx:.6f}")
        return rms, mx

    print("\n=== 原始 HR 音频统计 ===")
    audio_stats(hr_batch, "hr_audio")

    # 3) 通过底层 autoencoder 模型拿到 raw latent（尚未除以 scale）
    #    AutoencoderPretransform.encode 内部逻辑：
    #      encoded = model.encode_audio(...)
    #      return encoded / self.scale
    #    因此这里直接调用 encode_audio 拿到 raw latent。
    pre_model = getattr(pre, "model", None)
    if pre_model is None:
        raise RuntimeError("pretransform 没有 .model 属性，可能不是 AutoencoderPretransform。")

    pre_model.to(device)

    with torch.no_grad():
        raw_latent = pre_model.encode_audio(
            hr_batch,
            chunked=getattr(pre, "chunked", False),
            iterate_batch=getattr(pre, "iterate_batch", False),
        )  # [B, C_latent, L_latent]

    raw_std = raw_latent.std().item()

    with torch.no_grad():
        encoded_latent = pre.encode(hr_batch)  # raw / scale
    encoded_std = encoded_latent.std().item()

    print("\n=== Latent 统计量 ===")
    print(f"raw_latent std           = {raw_std:.6f}  (模型实际输出) ")
    print(f"encoded_latent std       = {encoded_std:.6f}  (raw/scale, 当前用于扩散模型的 latent)")
    print(f"当前 scale               = {scale:.6f}")
    if scale != 0.0:
        print(f"raw_std / scale          = {raw_std/scale:.6f}  (理论上应 ≈ encoded_std)")

    # 理想情况：希望传给 DiT 的 latent 标准差约为 1，则推荐：scale ≈ raw_std
    print(f"\n建议的 pretransform.scale 以使 latent std ≈ 1: {raw_std:.6f}")

    # 4) 解码不同尺度的 latent，看音频幅度是否被放大
    print("\n=== 不同 latent 下解码的音频幅度对比 ===")
    with torch.no_grad():
        # 4.1 encode -> decode（近似自编码重构）
        recon = pre.decode(encoded_latent)

        # 4.2 从 N(0,1) 采样 latent（模拟“扩散模型学到 std≈1 的 latent” 情况）
        unit_latent = torch.randn_like(encoded_latent)
        audio_unit = pre.decode(unit_latent)

        # 4.3 从 N(0, encoded_std) 采样 latent（与当前训练时的 latent 尺度匹配）
        matched_latent = torch.randn_like(encoded_latent) * encoded_std
        audio_matched = pre.decode(matched_latent)

    audio_stats(recon, "recon_from_encode")
    audio_stats(audio_unit, "decode_N(0,1)")
    audio_stats(audio_matched, "decode_N(0,latent_std)")

    print("\n说明：")
    print("- 若 `decode_N(0,1)` 的 rms / max 明显大于 `hr_audio`，则说明：")
    print("  扩散模型若输出 std≈1 的 latent，在当前 scale 下确实会导致音频幅度放大。")
    print("- 若 raw_latent std 很小（例如 ≈0.1），而 scale=1.0，则可以将 pretransform.scale 设为该 std，")
    print("  这样 encode 时 latent 会被放大到单位方差，decode(encode(x)) 仍然还原原始幅度，但扩散模型")
    print("  看到的 latent 尺度就与 N(0,1) 噪声一致，不会再因尺度不匹配导致 9dB LSD / 爆音问题。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug VAE latent scale for Stable Audio / SAGA-SR")
    parser.add_argument("--model_config", type=str, required=True, help="模型配置 JSON 路径，例如 saga_model_config_small.json")
    parser.add_argument("--audio_dir", type=str, required=True, help="用于统计的高分辨率音频目录")
    parser.add_argument("--num_examples", type=int, default=8, help="用于统计的样本数量")
    parser.add_argument("--duration", type=float, default=1.48, help="每条音频的目标时长（秒），需与训练时一致")

    args = parser.parse_args()

    compute_vae_scale(
        model_config_path=args.model_config,
        audio_dir=args.audio_dir,
        num_examples=args.num_examples,
        duration=args.duration,
    )
