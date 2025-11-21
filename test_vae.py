import json
import os

import torch
import torchaudio
from einops import rearrange
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict


def load_local_model(model_dir: str):
    config_path = os.path.join(model_dir, "model_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)

    for filename in ("model.safetensors", "model.ckpt"):
        ckpt_path = os.path.join(model_dir, filename)
        if os.path.exists(ckpt_path):
            state_dict = load_ckpt_state_dict(ckpt_path)
            model.load_state_dict(state_dict)
            break
    else:
        raise FileNotFoundError("No checkpoint file found in local model directory")

    return model, model_config


# 1. 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 从本地文件夹加载预训练模型
model, model_config = load_local_model("./stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

# ========== 模式选择 ==========
# 模式1: 纯文本生成
MODE = "text_only"  # 可选: "text_only", "audio_only", "text_and_audio"

# 音频输入文件路径（当使用audio_only或text_and_audio模式时）
INPUT_AUDIO_PATH = "./p360_002_8k.wav"  # 替换为你的音频文件路径

# init_noise_level控制对输入音频的修改程度: 0.0(几乎不变) 到 1.0(完全重新生成)
INIT_NOISE_LEVEL = 0.5  # 推荐范围: 0.3-0.8

# 3. 设置条件
if MODE == "text_only":
    # 纯文本模式
    conditioning = [{
        "prompt": "a woman says please call mike",
        "seconds_start": 0,
        "seconds_total": 3
    }]
    init_audio = None
    
elif MODE == "audio_only":
    # 纯音频模式（音频变化）
    conditioning = [{
        "prompt": "",  # 空提示或通用描述
        "seconds_start": 0,
        "seconds_total": 10
    }]
    # 加载输入音频
    audio_data, audio_sr = torchaudio.load(INPUT_AUDIO_PATH)
    init_audio = (audio_sr, audio_data)
    
elif MODE == "text_and_audio":
    # 音频+文本模式（文本引导的音频变化）
    conditioning = [{
        "prompt": "a female says 'please call stella'",  # 描述你想要的变化
        "seconds_start": 0,
        "seconds_total": 10
    }]
    # 加载输入音频
    audio_data, audio_sr = torchaudio.load(INPUT_AUDIO_PATH)
    init_audio = (audio_sr, audio_data)
else:
    raise ValueError(f"未知模式: {MODE}")

print(f"使用模式: {MODE}")
if init_audio is not None:
    print(f"输入音频: {INPUT_AUDIO_PATH}, 采样率: {init_audio[0]}Hz")
    print(f"噪声水平: {INIT_NOISE_LEVEL} (越高修改越大)")

# 4. 生成音频
output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device,
    init_audio=init_audio,  # 新增：音频输入
    init_noise_level=INIT_NOISE_LEVEL if init_audio is not None else 1.0  # 新增：噪声水平
)

# 5. 重排音频维度并保存
output = rearrange(output, "b d n -> d (b n)")
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

output_filename = f"output_{MODE}.wav"
torchaudio.save(output_filename, output, sample_rate)

print(f"音频已生成并保存为 {output_filename}")
