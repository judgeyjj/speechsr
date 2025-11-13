# SAGA-SR 复现实现总结

## 1. SAGA-SR 技术路线

### 1.1 核心架构
- **骨架模型**: DiT (Diffusion Transformer)
- **训练目标**: Conditional Flow Matching
- **编码器**: VAE（音频 ↔ latent 转换）

### 1.2 三种引导机制
1. **低分辨率音频引导**（主要声学）
   - `lr_latent` 通过 `input_concat_cond` 拼接到模型输入
   - 通道维度拼接，直接参与DiT计算

2. **频谱滚降点引导**（次要声学）
   - 计算：STFT时间轴聚合后计算roll-off（percentage=0.985）
   - 嵌入：标量 → Fourier Embeddings（sin/cos周期函数）
   - 注入：双通道
     - Cross-Attention：与文本嵌入拼接
     - Global：与时间步嵌入相加后prepend到DiT输入

3. **文本语义引导**
   - 使用Qwen2-Audio生成caption（训练用HR，推理用LR）
   - T5-base编码 → Cross-Attention注入

### 1.3 数据处理（论文标准）
- **低通滤波器生成低分辨率**
  - 4种滤波器：Chebyshev / Butterworth / Bessel / Elliptic
  - 截止频率：2-16kHz均匀随机
  - 滤波器阶数：2-10随机
  - 注意：截止频率≠采样率，滤波后仍保持44.1kHz

- **音频参数**
  - 采样率：44.1kHz
  - 音频长度：5.94秒 手动替换为1.48秒
  - 通道：单声道

### 1.4 推理策略
- **采样器**: Euler sampler（100 steps）
- **多重CFG**: 每步3次前向传播
  ```
  v_final = v_uncond + s_a*(v_acoustic - v_uncond) + s_t*(v_text - v_uncond)
  ```
  - s_a = 1.4（声学引导强度）
  - s_t = 1.2（文本引导强度）

### 1.5 训练参数
- 优化器：AdamW (β₁=0.9, β₂=0.999)
- 学习率：1.0e-5
- 批大小：256（可根据GPU调整）
- 训练步数：26,000

---

## 2. SAGA-SR 在 Stable Audio 上的扩展模块

### 2.1 Stable Audio 原有能力
- ✅ VAE编码器/解码器
- ✅ DiT扩散模型（支持`input_concat_cond`）
- ✅ T5文本条件器
- ✅ 数字条件（时间、采样率等）
- ✅ Flow Matching训练框架

### 2.2 SAGA-SR 新增模块

| 模块 | 原有 | 新增内容 | 文件 |
|------|------|----------|------|
| **数据生成** | ❌ | 4种低通滤波器（随机参数） | `dataset.py` |
| **Roll-off计算** | ❌ | 时间轴聚合、Fourier嵌入 | `spectral_features.py` |
| **Roll-off条件器** | ❌ | 双通道注入（Cross-Attn + Global） | `conditioner_rolloff.py` |
| **Caption生成** | ❌ | Qwen2-Audio适配器（训练/推理区分） | `audio_captioning_adapter.py` |
| **训练脚本** | ❌ | Flow Matching + 多条件整合 | `train_saga_sr.py` |
| **推理脚本** | ❌ | Euler采样 + 多重CFG（3次前向） | `inference_saga_sr.py` |
| **模型配置** | 部分 | 添加`input_concat_dim=64` | `saga_model_config.json` |

### 2.3 扩展方式
- **继承扩展**：`RolloffFourierConditioner` 独立实现，不修改Stable源码
- **配置扩展**：通过JSON配置添加新的conditioning类型
- **数据流扩展**：在训练/推理流程中整合新的条件

---

## 3. 代码实现完成度检查

### 3.1 Phase 1 - 基础架构 ✅
- ✅ `dataset.py`: 4种低通滤波器数据生成
- ✅ `saga_model_config.json`: `input_concat_dim=64` 配置
- ✅ `train_saga_sr.py`: `lr_latent`传入训练循环

### 3.2 Phase 2 - 核心创新 ✅
- ✅ `spectral_features.py`: Fourier Embeddings实现
  - `compute_spectral_rolloff()`: 时间轴聚合计算
  - `FourierEmbedding`: sin/cos周期嵌入
  - `SpectralRolloffProcessor`: 完整处理流程

- ✅ `conditioner_rolloff.py`: 双通道注入器
  - Cross-Attention路径实现
  - Global路径实现
  - 10% dropout支持

### 3.3 Phase 3 - 质量保证 ✅
- ✅ `inference_saga_sr.py`: 多重CFG推理
  - Euler采样器（100 steps）
  - 3次前向传播（无条件/仅声学/完整）
  - 论文公式组合（s_a=1.4, s_t=1.2）

- ✅ Roll-off计算细节
  - percentage=0.985
  - 时间轴聚合（非逐帧）

- ✅ 训练参数对齐
  - AdamW (lr=1e-5)
  - InverseLR scheduler
  - Flow Matching目标

### 3.4 Phase 4 - 增强功能 ✅
- ✅ `audio_captioning_adapter.py`: Qwen2-Audio集成
  - 本地模型支持
  - API调用支持
  - 训练/推理区分（use_hr_audio参数）
  - Caption缓存机制

---

## 4. 关键技术点对照表

| 论文要求 | 实现位置 | 状态 |
|----------|----------|------|
| 4种低通滤波器 | `dataset.py:_apply_lowpass_filter()` | ✅ |
| 截止频率2-16kHz | `dataset.py:116` | ✅ |
| 滤波器阶数2-10 | `dataset.py:117` | ✅ |
| STFT参数(2048/512) | `spectral_features.py:19` | ✅ |
| Roll-off 0.985 | `spectral_features.py:20` | ✅ |
| 时间轴聚合 | `spectral_features.py:33` | ✅ |
| Fourier嵌入 | `spectral_features.py:FourierEmbedding` | ✅ |
| 双通道注入 | `conditioner_rolloff.py:86-96` | ✅ |
| input_concat_dim=64 | `saga_model_config.json:88` | ✅ |
| Flow Matching | `train_saga_sr.py:95-102` | ✅ |
| 多重CFG (3次前向) | `inference_saga_sr.py:172-223` | ✅ |
| s_a=1.4, s_t=1.2 | `inference_saga_sr.py:225` | ✅ |
| AdamW优化器 | `train_saga_sr.py:137` | ✅ |
| lr=1e-5 | `train_saga_sr.py:138` | ✅ |
| Qwen2-Audio | `audio_captioning_adapter.py` | ✅ |
| 训练用HR caption | `audio_captioning_adapter.py:49` | ✅ |
| 推理用LR caption | `audio_captioning_adapter.py:50` | ✅ |

---

## 5. 使用流程

### 5.1 数据准备
```bash
# 将高分辨率音频放到目录
mkdir -p dataset/train/high_res
# 数据集会自动应用低通滤波生成低分辨率
```

### 5.2 预生成Caption（可选）
```bash
python audio_captioning_adapter.py
# 或在代码中：
# from audio_captioning_adapter import pregenerate_captions
# pregenerate_captions('dataset/train/high_res', mode='local')
```

### 5.3 训练
```bash
python train_saga_sr.py \
  --train_dir dataset/train/high_res \
  --model_config saga_model_config.json \
  --batch_size 4 \
  --max_steps 26000 \
  --use_caption  # 可选
```

### 5.4 推理
```bash
python inference_saga_sr.py \
  --input low_res_audio.wav \
  --output high_res_audio.wav \
  --checkpoint outputs/saga_sr_final.ckpt \
  --config saga_model_config.json \
  --target_rolloff 16000 \
  --num_steps 100 \
  --use_caption  # 可选
```

---

## 6. 总结

### ✅ 已完成
- 所有7个核心代码文件
- 严格对齐论文标准
- 完整的训练和推理流程
- 继承扩展方式（不修改Stable源码）

### 📝 技术特点
- **数据真实性**: 4种滤波器模拟真实降质场景
- **特征表达**: Fourier嵌入增强频率特征表达
- **引导强度**: 双通道注入确保引导信号充分
- **生成质量**: 多重CFG保证推理质量
- **可扩展性**: 模块化设计便于集成和修改

### 🎯 下一步
1. 准备训练数据（建议3800小时多域音频）
2. 根据GPU显存调整batch_size
3. 开始训练（预计48-72小时/RTX 3090）
4. 验证超分辨率效果
