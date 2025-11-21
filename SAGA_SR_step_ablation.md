# SAGA-SR 小模型逐步消融验证计划（Flow/EMA 对齐之后）

> 目的：不是找“最终瓶颈”，而是**系统地验证：哪些条件 / 设计对超分效果有显著影响**。
>
> 思路：从一个尽量简单、已经对齐官方 Rectified Flow + EMA 的 small 基线开始，
> 逐步只改很小的一处配置，每一步都保证 **不会引入维度错误**，并观察：
> - 损失曲线（训练 / 验证）；
> - LSD / SI-SDR 等指标；
> - 主观听感。

---

## 0. 前提与约定

- 代码基于当前 `speechsr/` 仓库，已经：
  - 在 `train_saga_sr.py` 中使用 Stable Audio 的 `rectified_flow` 公式和 EMA ；
  - `saga_model_config_small.json` 已设置：
    - `diffusion.diffusion_objective = "rectified_flow"`
    - `training.use_ema = true`
- 默认采样率 `44100`，VAE latent 维度 64，small DiT 配置：
  - `embed_dim = 512`, `depth = 8`, `num_heads = 8`
  - `io_channels = 64`, `input_concat_dim = 64`
  - `cond_token_dim = 512`（由 T5-base 768 维通过 conditioner 线性投影得到），`prepend_cond_dim = 512`
  - `global_cond_dim = 1024`（两个 number conditioner：`seconds_start/seconds_total` 各 512 维拼接）

**维度安全前提（small 配置）**：

- T5-base 的隐藏维度 = 768，经 T5Conditioner 投影成 `cond_dim = 512`，再作为 `cond_token_dim = 512` 输入 DiT cross-attn；
- small DiT 的 embedding 维度 = 512，与 `prepend_cond_dim = 512` 一致；
- `global_cond_dim = 1024 = cond_dim * len(global_cond_ids) = 512 * 2`，对应 `seconds_start/seconds_total` 两个 number conditioner 的拼接，再由 `to_global_embed` 投影到 `embed_dim = 512`；
- VAE latent 维度 = 64，与 `io_channels = 64`、`input_concat_dim = 64` 一致；
- `RolloffFourierConditioner` 的全局嵌入维度由 `embed_dim` 决定，可以同时适配 small / full；其 cross-attn 维度在 small 中应与 `cond_token_dim` 一致（见 Step 2 的说明）。

在下面每一步中，只要不改上述这些核心维度，就不会触发 shape mismatch。

---

## Step 0：当前 small 基线（已完成）

> **目的：** 给后续所有改动一个基准点。
>
> **结构：** Rectified Flow + EMA + 小 DiT + LR latent + 时间长度；无文本 cross-attn，无 rolloff。

### 0.1 配置与代码

- 使用 `saga_model_config_small.json`，保持原样：
  - `diffusion.cross_attention_cond_ids = []`
  - `diffusion.global_cond_ids = ["seconds_start", "seconds_total"]`
  - `diffusion.config.input_concat_dim = 64`
- `train_saga_sr.py`：
  - 数据集构造：
    ```python
    train_dataset = SAGASRDataset(
        ...,
        compute_rolloff=False,
        ...,
    )
    val_dataset = SAGASRDataset(
        ...,
        compute_rolloff=False,
        ...,
    )
    ```
  - Trainer 初始化：默认 `use_caption = True / False` 对 DiT 无影响（因为 cross-attn 被关掉）。
  - 训练时：
    - `lr_latent` 通过 `input_concat_cond` 注入 DiT；
    - `seconds_start/seconds_total` 作为 global cond。

### 0.2 建议记录

- 记录：
  - 训练 loss 曲线；
  - `val/lsd`, `val/si_sdr`；
  - 若有时间，导出若干样本的主观听感评估。

后续所有 Step 都以此为对比基准。

---

## Step 1：在 small 上打开文本条件（LR latent + 文本，无 rolloff）

> **目的：** 验证“文本语义条件本身”对超分的影响，仍然不引入 rolloff。
>
> **只改一处 JSON，不会引入维度错误**。

### 1.1 修改内容

**文件：** `saga_model_config_small.json`

1. 在 `"diffusion"` 段中：

   将：
   ```json
   "cross_attention_cond_ids": [],
   ```
   改为：
   ```json
   "cross_attention_cond_ids": ["prompt"],
   ```

   - 维度安全说明：
     - 条件 `"prompt"` 对应 `conditioning.configs` 里的 T5 条目（T5-base 输出 768 维），经 T5Conditioner 线性投影到 `cond_dim = 512`；
     - `diffusion.config.cond_token_dim = 512`，与 `cond_dim` 一致，且 `embed_dim=512, num_heads=8` → `dim_heads=64`，cross-attn 的 `num_heads == kv_heads == 8`，不会出现 head 数不一致；
     - `diffusion.config.global_cond_dim = 1024 = 512 * 2`，对应 `seconds_start/seconds_total` 两个 number conditioner 的拼接，随后由 `to_global_embed` 投影到 `embed_dim=512`；
     - 这些 small 模型的维度配置是从 full 配置（`cond_dim=768, cond_token_dim=768, global_cond_dim=1536`）等比例缩放而来，目的是避免 cross-attn / global 条件在 Stable Audio 的 attention 中触发 `mat1 @ mat2` 维度错误。

2. 训练命令行：

   - 如果你希望使用 Qwen caption + T5：
     - 不加 `--disable_caption`（默认 `use_caption=True`）。
   - 如果你只想先用 transcript（数据集自带文本）而不调用 Qwen：
     - 可以显式加 `--disable_caption`，这样仅使用 `metadata['transcript']`。

### 1.2 预期与观察

- 对比 Step 0：
  - 观察训练收敛速度是否变快；
  - 验证指标（尤其是语音清晰度、是否更符合文本语义）。
- 如果 Step 1 基本没有提升，说明“只加文本”对你的当前数据/任务帮助有限，后续需要重点看 rolloff / 大模型。

---

## Step 2：在 small 上启用 rolloff 条件（LR latent + 文本 + rolloff）

> **目的：** 验证“频谱 rolloff 条件”对高频重建质量的影响。
>
> **只改 DataLoader 的 `compute_rolloff`，不动维度相关配置。**

### 2.1 修改内容

**文件：** `train_saga_sr.py`

1. 训练数据集：

   将：
   ```python
   train_dataset = SAGASRDataset(
       audio_dir=args.train_dir,
       sample_rate=44100,
       duration=1.48,
       compute_rolloff=False,
       num_samples=65536,
       audio_channels=audio_channels,
   )
   ```
   改为：
   ```python
   train_dataset = SAGASRDataset(
       audio_dir=args.train_dir,
       sample_rate=44100,
       duration=1.48,
       compute_rolloff=True,
       num_samples=65536,
       audio_channels=audio_channels,
   )
   ```

2. 验证数据集同样改为 `compute_rolloff=True`：

   ```python
   val_dataset = SAGASRDataset(
       audio_dir=args.val_dir,
       sample_rate=44100,
       duration=1.48,
       compute_rolloff=True,
       num_samples=65536,
       audio_channels=audio_channels,
   )
   ```

3. 其它代码保持不变，但 **small 模型在启用 rolloff 时需要额外对齐一个维度参数**：

   - `SAGASRDataset` 会在 metadata 中填入 `rolloff_low` / `rolloff_high`；
   - `training_step` 中已有：
     ```python
     if 'rolloff_low' in metadata[0] and 'rolloff_high' in metadata[0]:
         rolloff_low = torch.stack([m['rolloff_low'] for m in metadata])
         rolloff_high = torch.stack([m['rolloff_high'] for m in metadata])
         rolloff_cond = self.rolloff_conditioner(rolloff_low, rolloff_high, apply_dropout=True)
     ```
   - `RolloffFourierConditioner` 的维度设置，**small 情况下建议改为**：
     ```python
     self.rolloff_conditioner = RolloffFourierConditioner(
         embedding_dim_cross=512,  # small: 对齐 cond_token_dim = 512
         embedding_dim_global=config['model']['diffusion']['config']['embed_dim'],
         dropout_rate=0.1,
     )
     ```
     - 对 small：`embed_dim = 512`，则 rolloff global = [B, 512]，与 DiT 的 `embed_dim` 匹配；
     - cross-attn 分支输出 [B, 1, 512]，与 small 配置下的 `cond_token_dim = 512` 一致；
     - 对 full：可以保持 `embedding_dim_cross=768`，与 full 配置里的 `cond_token_dim = 768` 对齐。
   - `_build_rolloff_prepend` 用 `embed_dim` 作为 prepend 维度：
     ```python
     rolloff_prepend = rolloff_global.unsqueeze(1) + timestep_embed.unsqueeze(1)  # [B, 1, embed_dim]
     ```
     对 small：embed_dim=512，与 `prepend_cond_dim = 512` 完全一致。

**因此：这一改动不会引入任何维度错误。**

### 2.2 预期与观察

- 与 Step 1 对比，主要看：
  - 高频细节是否更自然、更少“过度锐化 / 噪声”；
  - LSD 是否显著下降（特别是高频段）；
  - 若你在评估中使用低频替换，rolloff 条件应有正向影响。

---

## Step 2′：在 small 上排查 rolloff + 低频替换带来的谱图伪影（亮线）

> **背景：** 在 Step 2 中启用 rolloff 后，如果同时开启了验证阶段的「低频替换」 `_low_frequency_replace`，谱图上可能在 `rolloff_low` 附近出现一条稳定的横向亮线。这更像是**后处理在频域的硬拼接边界**，而非模型本身学到的伪音。
>
> **目标：** 将「rolloff 条件」与「低频替换」解耦，先确认亮线是否由 `_low_frequency_replace` 导致。

### 2′.1 操作：在 small 上关闭低频替换，只保留 rolloff 条件

- **命令行：** 在 Step 2 的基础上，增加：

  ```bash
  python train_saga_sr.py \
      --config saga_model_config_small.json \
      ...其它参数... \
      --disable_val_lowfreq_replace
  ```

  - 训练配置保持 Step 2 完全一致：
    - `saga_model_config_small.json` 中仍然：
      - `embed_dim = 512, num_heads = 8`
      - `cond_dim = 512, cond_token_dim = 512`
      - `global_cond_dim = 1024`（两个 number conditioner 各 512 维拼接）
    - `train_saga_sr.py` 中仍然：
      - `rolloff_conditioner = RolloffFourierConditioner(embedding_dim_cross=512, embedding_dim_global=embed_dim, ...)`
      - 数据集 `compute_rolloff=True`。
  - 新增的 `--disable_val_lowfreq_replace` 只会影响 `validation_step` 里的：

    ```python
    if not getattr(self.hparams, "disable_val_lowfreq_replace", False) and rolloff_low is not None:
        pred_audio = self._low_frequency_replace(pred_audio, lr_audio_mono, rolloff_low)
    ```

    变成「条件为 False」，从而**完全跳过低频替换**这一步。

- **维度安全说明：**

  - 这一操作不改变任何张量的通道数或形状，仅仅是「少做一次频域替换」；
  - small 配置下所有维度关系（`embed_dim / cond_token_dim / global_cond_dim / input_concat_dim` 等）保持与 Step 2 完全一致，因此不会引入新的 shape mismatch。

### 2′.2 观察要点

- 与 Step 2（启用 rolloff + 低频替换）的结果对比：
  - **谱图上那条固定亮线是否消失或显著减弱；**
  - 高频整体形态是否变化不大（说明 rolloff 条件仍在起作用，只是去掉了硬拼接）；
  - LSD / SI-SDR 的变化（通常关闭低频替换后，指标可能略有波动，但更能反映纯模型的表现）。

- 若：
  - 亮线在 Step 2′ 中基本消失，而其它高频结构保留 → 可以**基本确认伪影来自 `_low_frequency_replace` 的硬阈值边界**；
  - 亮线仍然存在，则需要进一步排查 rolloff 条件本身（但按当前实现更不太可能）。

### 2′.3 后续建议（可选）

- 如果确认是低频替换导致伪影，但你仍然希望使用「低频保护」：

  - 可以在未来版本中将 `_low_frequency_replace` 改成**带过渡带的 cross-fade**，而不是简单的：
    ```python
    mask = freqs.unsqueeze(0) <= cutoff
    gen_fft = torch.where(mask, lr_fft, gen_fft)
    ```
  - 示例思路（仅逻辑）：
    - 设 `f1 = 0.9 * cutoff`, `f2 = 1.1 * cutoff`；
    - `f < f1` 用 100% LR；
    - `f1~f2` 按频率线性或余弦插值；
    - `f > f2` 用 100% 生成谱。
  - 这类修改同样不会改变任何张量维度，只是频域混合策略更平滑，可在确认问题后再实现。

---

## Step 3：在 small 上加入 LR latent 的 10% dropout（CFG 风格）

> **目的：** 模拟论文中对 LR latent 做 10% dropout，引入一种“对 LR 依赖的 CFG”效果，观察是否能改善泛化/听感。
>
> **只改一个标量超参数，不动任何张量维度。**

### 3.1 修改内容

**文件：** `train_saga_sr.py`

1. 在 `SAGASRTrainer.__init__` 中：

   将：
   ```python
   self.latent_dropout_prob = 0
   ```
   改为：
   ```python
   self.latent_dropout_prob = 0.1
   ```

2. 其余 dropout 逻辑已写好：

   ```python
   def _apply_latent_dropout(self, latent: torch.Tensor, training: bool) -> torch.Tensor:
       if not training or self.latent_dropout_prob <= 0.0:
           return latent

       if self.latent_dropout_prob >= 1.0:
           return torch.zeros_like(latent)

       keep_mask = (torch.rand(latent.shape[0], device=latent.device) >= self.latent_dropout_prob)
       ...
       return latent * keep_mask
   ```

   - 这里所有操作都是 element-wise 掩码乘法；
   - 不会改变 `latent` 的形状或通道数。

### 3.2 预期与观察

- 对比 Step 2：
  - 训练初期可能略微不稳定，但中后期若有正向效果，可稍微提升泛化；
  - 对完全 overfit 的小数据实验，可能提升有限，但这一步是为后续 full 模型铺路。

---

## Step 4：从 small 切换到 full 模型时的注意点（可选）

> 当前重点还是 small 消融。这里列出以后切到大模型时如何“只调大 DiT，其它设置沿用”的要点。

### 4.1 使用 `saga_model_config.json` 作为 full 配置

- 与 small 相比，主要差异：
  - `embed_dim = 1536`, `depth = 24`, `num_heads = 24`；
  - `prepend_cond_dim = 1536`；
  - `global_cond_dim = 1536`；
  - 其他如 `io_channels = 64`, `input_concat_dim = 64`, `cond_token_dim = 768` 保持一致。

### 4.2 可以安全复用的改动

你在 Step 1–3 中对 small 做的所有改动，都可以 **一模一样** 地迁移到 `saga_model_config.json` / `train_saga_sr.py` 中，而不会引起维度错误：

1. **文本条件：**
   - full config 原本就有：
     ```json
     "cross_attention_cond_ids": ["prompt", "seconds_start", "seconds_total"]
     ```
   - `cond_token_dim = 768` 与 T5-base 一致，因此文本条件维度安全。

2. **rolloff 条件：**
   - `RolloffFourierConditioner` 中 `embedding_dim_global` 会变成 1536，与 full DiT `embed_dim=1536` 匹配；
   - cross-attn 分支仍然是 768 维，与 `cond_token_dim = 768` 一致；
   - prepend 分支是 [B, 1, 1536]，与 `prepend_cond_dim = 1536` 一致。

3. **latent dropout：**
   - 仍然只是对 shape=[B, 64, L] 的 LR latent 乘一个 [B,1,1] 的 mask，不改变维度。

**因此：当你从 small 切回 full 模型时，只要不改这些核心维度参数（embed_dim / cond_token_dim / latent_dim等），Step 1–3 的所有设置都是维度安全的。**

---

## 5. 建议的实验顺序与记录方式

**建议顺序：**

1. **Step 0 → Step 1：** small，LR-only → LR + 文本；
2. **Step 1 → Step 2：** small，LR + 文本 → LR + 文本 + rolloff；
3. **Step 2 → Step 3：** small，LR + 文本 + rolloff → 再加 LR dropout 10%；
4. 若 small 上已经能明显看到 “哪个条件最关键”，再选择性地把该条件迁移到 full 模型上验证。

**每一步建议记录：**

- 训练/验证的 loss 曲线（同一纵轴方便比较）；
- `val/lsd`, `val/si_sdr` 的数值变化；
- 至少若干条样本的 A/B 听感对比：
  - Step N vs Step N+1；
  - 特别关注：高频细节、伪影、连续性。

这样你可以非常清楚地回答：

- 仅有 LR latent 时，模型大致能做到什么程度；
- 加入文本语义后，有没有补上“语义先验”；
- rolloff 这个 acoustic 条件，对高频 / 失真有什么实质影响；
- LR dropout（CFG 风格）对最终 SR 质量的作用。

最后，如果你在某一步看到明显的质变（例如 Step 1 或 Step 2），我们就可以再基于那一步，进一步做更细的分析（比如换 timesteps、调 CFG 比例、只在推理端使用 EMA 等）。
