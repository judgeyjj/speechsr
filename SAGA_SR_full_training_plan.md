# SAGA-SR 大模型训练计划（基于 `saga_model_config.json`）

> 目标：在已经完成 small 模型的消融之后，**正式切换到 full 大模型**（`saga_model_config.json`），
> 并且按照“逐步增加复杂度”的方式训练：
>
> 1. 先只用 LR latent（+ 时间条件），确认大模型本身的收敛与记忆能力；
> 2. 再加入文本条件（T5）；
> 3. 视需要在容量足够的前提下，尝试 rolloff 条件；
> 4. 全程可以选择是否使用低频替换（现在支持在不计算 rolloff 的情况下，仅根据数据集的低通截止频率做低频替换）。
>
> 本文档只描述 **full 模型** 的训练步骤，不再讨论 small 扩展。

---

## 0. 配置与维度前提（full）

- 使用配置文件：`saga_model_config.json`
- 核心维度：

  - **VAE / latent：**
    - `latent_dim = 64`
    - `model.io_channels = 64`
  - **DiT（full）：**
    - `embed_dim = 1536`
    - `depth = 24`
    - `num_heads = 24` → `dim_head = 1536 / 24 = 64`
  - **条件维度：**
    - `conditioning.cond_dim = 768`
    - `diffusion.config.cond_token_dim = 768`
    - `diffusion.config.global_cond_dim = 1536 = cond_dim * 2`
    - `diffusion.config.prepend_cond_dim = 1536`
  - **输入通道 / concat：**
    - `diffusion.config.io_channels = 64`
    - `diffusion.config.input_concat_dim = 64`

- 代码侧（`train_saga_sr.py`）：

  - `SAGASRTrainer.__init__` 中，已经改为自动对齐 rolloff conditioner 维度：

    ```python
    diff_cfg = config['model']['diffusion']['config']
    cond_token_dim = diff_cfg.get('cond_token_dim', config['model']['conditioning'].get('cond_dim'))
    self.rolloff_conditioner = RolloffFourierConditioner(
        embedding_dim_cross=cond_token_dim,      # full: 768；small: 512
        embedding_dim_global=diff_cfg['embed_dim'],  # full: 1536；small: 512
        dropout_rate=0.1,
    )
    ```

  - 因此，在 full 配置下：
    - rolloff cross-attn 输出 [B,1,768]，与 `cond_token_dim=768` 匹配；
    - rolloff global 输出 [B,1536]，通过 `_build_rolloff_prepend` 变成 [B,1,1536]，与 `prepend_cond_dim=1536`、`embed_dim=1536` 一致。

- **低频替换与 rolloff 的解耦：**

  - `SAGASRDataset` 现在会在生成 LR 时，把低通滤波器的**截止频率**写入 metadata：

    ```python
    lr_audio_np, cutoff_freq = self._apply_lowpass_filter(...)
    metadata = {
        'lr_audio': lr_audio,
        'audio_path': audio_path,
        'lr_cutoff_hz': float(cutoff_freq),
    }
    ```

  - 在验证阶段的 `validation_step` 中，低频替换逻辑改为：

    ```python
    cutoff_for_lfr = None
    if rolloff_low is not None:
        cutoff_for_lfr = rolloff_low
    elif 'lr_cutoff_hz' in metadata[0]:
        cutoff_for_lfr = torch.tensor([m['lr_cutoff_hz'] for m in metadata], ...)

    if not self.hparams.disable_val_lowfreq_replace and cutoff_for_lfr is not None:
        pred_audio = self._low_frequency_replace(pred_audio, lr_audio_mono, cutoff_for_lfr)
    ```

  - 这意味着：**即使不计算谱滚降（`--compute_rolloff` 不开），仍然可以用数据集中生成 LR 时的低通截止频率进行低频替换**。

  - CLI 行为也恢复为：

    ```bash
    --disable_val_lowfreq_replace  # 显式关闭低频替换（默认不开启此 flag → 默认启用低频替换）
    ```

---

## 1. Step F0：full，大模型 LR-only 基线

> 目的：在不引入文本 / rolloff 条件的前提下，验证 full DiT + LR latent 的收敛情况和容量上限。

### 1.1 JSON 修改（full 配置）

在 `saga_model_config.json` 中，仅做一处改动以关闭文本 cross-attn：

```jsonc
"diffusion": {
  "type": "dit",
  "diffusion_objective": "rectified_flow",
- "cross_attention_cond_ids": ["prompt", "seconds_start", "seconds_total"],
+ "cross_attention_cond_ids": [],
  "global_cond_ids": ["seconds_start", "seconds_total"],
  "config": {
    "io_channels": 64,
    "input_concat_dim": 64,
    "embed_dim": 1536,
    "depth": 24,
    "num_heads": 24,
    "cond_token_dim": 768,
    "global_cond_dim": 1536,
    "prepend_cond_dim": 1536,
    "project_cond_tokens": false,
    "transformer_type": "continuous_transformer"
  },
  ...
}
```

- 只改 `cross_attention_cond_ids`，不改任何维度。
- DiT 仍然接收：
  - LR latent（`input_concat_cond`）；
  - global 条件（时间长度：`seconds_start/seconds_total`）。

### 1.2 训练命令示例

- 不启用 rolloff，不做频谱滚降相关 conditioning：

```bash
python train_saga_sr.py \
    --train_dir /path/to/train_hr \
    --val_dir /path/to/val_hr \
    --model_config saga_model_config.json \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_steps 26000 \
    --disable_caption \
    --disable_val_lowfreq_replace  # 可选：若你想先看“纯模型”，可以暂时关掉低频替换
```

- 观测：
  - train/val loss 收敛曲线；
  - `val/lsd`, `val/si_sdr`；
  - 几条样本的主观听感，关注：
    - 高频是否比 small 有明显改善；
    - 是否出现崩坏 / 不收敛迹象。

---

## 2. Step F1：full + 文本条件（LR + 文本）

> 目的：在 full 容量下，引入 T5 文本条件，观察对语义对齐和细节的影响。

### 2.1 JSON 修改

在 `saga_model_config.json` 中，把刚才关闭的 cross-attn 打开：

```jsonc
"diffusion": {
  "type": "dit",
  "diffusion_objective": "rectified_flow",
- "cross_attention_cond_ids": [],
+ "cross_attention_cond_ids": ["prompt"],
  "global_cond_ids": ["seconds_start", "seconds_total"],
  ...
}
```

- 与 small 一样，`prompt` 对应的 T5 conditioner 输出 768 维，通过 `cond_dim=768` → `cond_token_dim=768` 输入 DiT；
- 时间条件仍然只走 global 分支，不再通过 cross-attn（避免 head 个数/维度混乱）。

### 2.2 训练命令示例

- 打开文本（可选用 caption / transcript）：

```bash
python train_saga_sr.py \
    --train_dir /path/to/train_hr \
    --val_dir /path/to/val_hr \
    --model_config saga_model_config.json \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_steps 26000 \
    # 根据需要是否启用 Qwen caption：
    #   - 默认 use_caption=True，会使用 caption + transcript
    #   - 加 --disable_caption 则只用 transcript
    # --disable_caption \
    --disable_val_lowfreq_replace  # 初期建议先关掉低频替换，方便观察纯模型
```

- 对比 Step F0：
  - 看 train/val loss 是否更快下降；
  - 语音内容与文本的一致性是否明显提升；
  - 高频伪影是否增多或减少。

---

## 3. Step F2：full + 文本 + 可选 rolloff（仅在 small/Step F1 表现良好后）

> 目的：在 full 容量足够的前提下，再次谨慎地引入 rolloff 条件，观察它对高频纹理的影响。

### 3.1 启用 rolloff 特征

1. **命令行：** 打开 `--compute_rolloff`，但可以先 **关闭低频替换**，只看 rolloff 条件本身：

   ```bash
   python train_saga_sr.py \
       ...同 Step F1... \
       --compute_rolloff \
       --disable_val_lowfreq_replace
   ```

2. **代码侧维度：**

   - 已经由 `RolloffFourierConditioner` 自动对齐：
     - cross-attn 分支：768 维，拼到 T5 文本 tokens 后；
     - global 分支：1536 维，通过 `_build_rolloff_prepend` 生成 [B,1,1536] 的 prepend token。

3. **观测点：**

   - VS Step F1：
     - 高频是否更自然 / 少伪影；
     - 谱图是否出现新的结构性 artefact；
     - LSD，特别是高频段的变化。

### 3.2 在不开 rolloff 时启用低频替换（可选）

> 你之前已经验证：谱滚降条件本身对语音高频存在风险，但“低频替换”本身是有效的后处理。现在我们支持：**在不开启 rolloff 特征的情况下，仍然可以根据 LR 低通截止频率做低频替换**。

- 若 **不想要 rolloff 条件**，但想要 low-frequency replace：

  ```bash
  python train_saga_sr.py \
      ...同 Step F0/F1... \
      # 不加 --compute_rolloff  => 不计算 rolloff_low / rolloff_high，不参与条件
      --disable_val_lowfreq_replace   # 去掉这个 flag 或不加，就会启用基于 lr_cutoff_hz 的低频替换
  ```

  - 因为现在：
    - 数据集总是保存 `lr_cutoff_hz`（生成 LR 时的低通截止频率）；
    - 验证阶段会优先使用 rolloff_low（若有），否则回退到 `lr_cutoff_hz`。

- 若想完全关闭低频替换：

  ```bash
  python train_saga_sr.py ... --disable_val_lowfreq_replace
  ```

---

## 4. 建议的 full 模型训练顺序

1. **Step F0：full + LR-only（无文本、无 rolloff）**
   - 确认：
     - 大模型在小数据上的“记忆能力”是否已经足够；
     - 是否存在明显的训练不稳定 / 爆炸。

2. **Step F1：full + LR + 文本**
   - 在确认收敛的前提下再加文本条件；
   - 对比：语义对齐程度 / 高频表现。

3. **Step F2（可选）：full + LR + 文本 + rolloff（或仅 LFR）**
   - 优先尝试“不开 rolloff 只开低频替换”；
   - 若低频替换 + 大模型表现良好，再谨慎尝试 rolloff 条件本身。

4. **若 full 上已经得到满意结果：**
   - 再考虑：
     - LR latent dropout（CFG 风格）；
     - 采样步数、t-schedule、CFG 比例等细节调优。

通过以上步骤，你可以在 **大模型设定** 下，把 small 上得到的经验（特别是文本 / rolloff / 低频替换的影响）系统地迁移过来，并且始终保持维度安全。
