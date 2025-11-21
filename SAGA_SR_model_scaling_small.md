# SAGA-SR 小模型容量扩展实验（基于 `saga_model_config_small.json`）

> 目标：在 **不一次性跳到超大模型** 的前提下，逐步增大 DiT 容量，观察：
>
> - 是否能更好地“记住”小数据集；
> - 对超分主观听感 / LSD / SI-SDR 的影响；
> - 在容量足够的前提下，再评估文本 / rolloff 条件各自的作用。
>
> 所有实验都基于同一个 JSON：`saga_model_config_small.json`，只改其中 DiT 相关字段，并保持 **维度严格自洽**，避免再次出现 shape mismatch。

---

## 0. 现有 small 配置回顾（起点）

- 文件：`saga_model_config_small.json`
- 关键字段：

  - **VAE / latent：**
    - `latent_dim = 64`（VAE bottleneck）
    - `model.io_channels = 64`
  - **DiT（small 起点）：**
    - `embed_dim = 512`
    - `depth = 8`
    - `num_heads = 8` → `dim_head = 512 / 8 = 64`
  - **输入通道 / concat：**
    - `diffusion.config.io_channels = 64`
    - `diffusion.config.input_concat_dim = 64` （LR latent 也是 64 维通道）
  - **条件相关维度（已在之前修正）：**
    - `conditioning.cond_dim = 512`
    - `diffusion.config.cond_token_dim = 512`
    - `diffusion.config.global_cond_dim = 1024` = `cond_dim * 2`（`seconds_start` + `seconds_total` 两个 number conditioner）
    - `diffusion.config.prepend_cond_dim = 512`

- **注意：**
  - T5-base 原始隐藏维度为 768，Stable Audio 的 `T5Conditioner` 会先把 768 投影到 `cond_dim=512`，再作为 cross-attn token (`cond_token_dim=512`) 输入 DiT；
  - global 条件（秒数）先各自投影到 512，再在 conditioner 里拼接成 [B, 1024]，由 DiT 的 `to_global_embed: Linear(1024 → embed_dim)` 投影到 512；
  - LR latent 维度始终是 64，与 `io_channels` / `input_concat_dim` 一致。

**结论：** 在 small 当前配置下，所有维度已经自洽，可以作为容量扩展实验的安全起点。

---

## 1. 实验 A：只加深，不加宽（512 维，depth 从 8 → 16）

> 目的：在维持单层计算量不变的前提下，通过增加层数提升模型表达能力，先看能否明显改善“记不住小样本”的问题。

### 1.1 JSON 修改

**文件：**`saga_model_config_small.json`

在 `"diffusion" → "config"` 段中：

```jsonc
{
  "io_channels": 64,
  "input_concat_dim": 64,

  "embed_dim": 512,
  "depth": 8,        // ← 修改这里
  "num_heads": 8,

  "cond_token_dim": 512,
  "global_cond_dim": 1024,
  "prepend_cond_dim": 512,
  "project_cond_tokens": false,
  "transformer_type": "continuous_transformer"
}
```

改为：

```jsonc
  "embed_dim": 512,
  "depth": 16,       // 由 8 提升到 16
  "num_heads": 8,
```

- **维度安全说明：**
  - 只增加层数，**不改任何维度参数**；
  - cross-attn：`embed_dim=512, num_heads=8, cond_token_dim=512` 保持不变；
  - global：`global_cond_dim=1024`，仍然由 `to_global_embed: 1024 → 512` 投影到 `embed_dim`；
  - VAE / input_concat 维度不变。

### 1.2 建议实验

- 训练配置基本沿用 small 基线（可先关 rolloff，仅用 `LR + 文本`）：
  - `--compute_rolloff` 可先不开；
  - `--disable_val_lowfreq_replace` 建议保持默认开启（True）。
- 对比：
  - A0：small 原始（512, depth=8）；
  - A1：small-deep（512, depth=16）。
- 重点观察：
  - 小数据集上的“记忆能力”（loss 是否更容易降到很低，甚至过拟合）；
  - 高频细节是否更连贯；
  - 训练稳定性 / 显存占用。

---

## 2. 实验 B：在变深的基础上适度变宽（768 维，depth=16）

> 目的：在 A1 仍不足以“记住小数据”的情况下，进一步增加每层宽度，但**不要一步跳到 full 的 1536 维**，先用中间档：`embed_dim = 768`。

这里我们选择：

- `embed_dim = 768`
- `num_heads = 12` → `dim_head = 768 / 12 = 64`
- `cond_dim = 768`
- `cond_token_dim = 768`
- `global_cond_dim = 1536` = `cond_dim * 2`

这样 cross-attn / global 条件都与 full 配置在**维度关系上对齐**，只是层数和宽度小一截。

### 2.1 JSON 修改

在 `saga_model_config_small.json` 中，做如下改动：

1. **conditioning 段：**

   ```jsonc
   "conditioning": {
     "configs": [
       ...
     ],
-    "cond_dim": 512
+    "cond_dim": 768
   }
   ```

2. **diffusion.config 段：**

   ```jsonc
   "diffusion": {
     "type": "dit",
     "diffusion_objective": "rectified_flow",
     "cross_attention_cond_ids": ["prompt"],
     "global_cond_ids": ["seconds_start", "seconds_total"],
     "config": {
-      "io_channels": 64,
-      "input_concat_dim": 64,
-
-      "embed_dim": 512,
-      "depth": 8,
-      "num_heads": 8,
-
-      "cond_token_dim": 512,
-      "global_cond_dim": 1024,
-      "prepend_cond_dim": 512,
+      "io_channels": 64,
+      "input_concat_dim": 64,
+
+      "embed_dim": 768,
+      "depth": 16,
+      "num_heads": 12,
+
+      "cond_token_dim": 768,
+      "global_cond_dim": 1536,
+      "prepend_cond_dim": 768,
       "project_cond_tokens": false,
       "transformer_type": "continuous_transformer"
     },
     ...
   }
   ```

- **维度安全说明：**
  - cross-attn：
    - `embed_dim = 768, num_heads = 12` → `dim_head = 64`；
    - `cond_token_dim = 768` → `kv_heads = 768 / 64 = 12`，与 `num_heads` 完全一致，不会触发 grouped-query 的 head 不整除问题；
  - conditioner：
    - `cond_dim = 768` → T5Conditioner 会将 768（T5-base 原始维度）投影到 768，本质上是 identity（或近似）；
    - number conditioner 会各生成 768 维，然后拼接成 `global_cond_dim = 1536`；
  - global：`to_global_embed: Linear(1536 → 768)`，输出与新的 `embed_dim` 对齐；
  - prepend：`prepend_cond_dim = 768`，与 `embed_dim` 一致，`_build_rolloff_prepend` 仍然返回 [B,1,embed_dim]；
  - VAE / input_concat 仍然保持 64，不受影响。

### 2.2 与 full 模型的关系

- full 配置（`saga_model_config.json`）的 DiT 是：
  - `embed_dim = 1536`, `depth = 24`, `num_heads = 24`
  - `cond_dim = 768`, `cond_token_dim = 768`, `global_cond_dim = 1536`
- 实验 B 的 768 维版本可以看作：
  - **保留 full 的条件维度关系（cond / global / prepend）不变；**
  - DiT 的宽度从 1536 减半到 768，深度从 24 降到 16；
  - 有利于在显存可控的前提下，先看“中等模型”是否已经足够好。

---

## 3. 使用方式与实验顺序建议

### 3.1 命令行示例（结合新的 `--compute_rolloff` 开关）

假设你只想做 **LR + 文本**，暂时关闭 rolloff 与低频替换：

```bash
python train_saga_sr.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --model_config saga_model_config_small.json \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --max_steps 26000 \
    --disable_val_lowfreq_replace \
    # 不加 --compute_rolloff  => 数据集不写入 rolloff_low/high
```

若要开启 rolloff 特征但仍关闭低频替换：

```bash
python train_saga_sr.py \
    ...同上... \
    --compute_rolloff \
    --disable_val_lowfreq_replace
```

### 3.2 建议的模型容量实验顺序

1. **A0（现有 small 基线）：**
   - `embed_dim = 512, depth = 8, num_heads = 8`
   - `cond_dim = 512, cond_token_dim = 512, global_cond_dim = 1024`
   - 建议：先在 **不带 rolloff、关闭低频替换** 的设置下，确保基线是“合理但明显偏小”的。

2. **A1（只加深）：**
   - `embed_dim = 512, depth = 16, num_heads = 8`（第 1 节）
   - 对比 A0：
     - 查看是否已经能明显“记住小样本”（train loss 迅速降到很低）；
     - 如果 A1 仍明显欠拟合，则说明需要进一步加宽。

3. **B1（加深 + 适度加宽到 768 维）：**
   - `embed_dim = 768, depth = 16, num_heads = 12`
   - `cond_dim = 768, cond_token_dim = 768, global_cond_dim = 1536, prepend_cond_dim = 768`
   - 与 full 在条件维度上对齐，但总参数量远小于 1536×24 的完整大模型。

4. **若 B1 已经足够：**
   - 可以在 B1 上再试：
     - 打开 `--compute_rolloff`，只看 rolloff 在“容量足够”时是正面还是负面；
     - 调整 LR latent dropout（`self.latent_dropout_prob = 0.1`）等细节。

5. **若 B1 仍不足：**
   - 再考虑：
     - 进一步加深（如 `depth = 24`）或加宽到 `embed_dim = 1024, num_heads = 16`；
     - 或者直接切换到官方 full 配置 `saga_model_config.json`，但这一步需要重新评估显存与训练时间成本。

---

## 4. 后续计划（建议）

1. **阶段一：容量 vs 记忆能力**
   - 依次跑：A0 → A1 → B1（均不带 rolloff，仅 `LR + 文本`，关闭低频替换）。
   - 记录：
     - train / val loss 曲线；
     - `val/lsd`, `val/si_sdr`；
     - 小数据上是否出现明显过拟合（有助于判断容量是否“足够大”。）

2. **阶段二：在“容量足够”的模型上重新审视 rolloff**
   - 选择 A1 或 B1 中效果最好的一档模型：
     - 对比 `--compute_rolloff` 关 / 开 时的谱图与指标；
     - 确认 rolloff 对语音高频纹理是否仍然有“空洞”风险。

3. **阶段三：若需要，再考虑 full 模型或更大宽度**
   - 若 B1 上 rolloff 的表现依旧不佳，可以在大模型上：
     - 改进 rolloff 的定义（例如不再用单一标量、或区分语音 / 非语音段）；
     - 或者只在推理端做 rolloff 相关的后处理，而不在训练中强约束。

通过上述步骤，你可以在 **完全控制维度安全** 的前提下，逐步找到一个“既能记住小样本，又不会因容量不足/条件过强导致伪影”的模型规模。
