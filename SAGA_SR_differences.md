# SAGA-SR 论文与当前实现差异对比（假设使用完整 JSON 配置）

> 本文档对比的是：  
> - **论文设定**：`SAGA-SR: Semantically and Acoustically Guided Audio Super-Resolution`  
> - **当前实现**：你仓库 `speechsr/` 下的代码（特别是 `SAGA_SR_repro.md`、`dataset.py`、`train_saga_sr.py`、`inference_saga_sr.py`、`saga_sampling.py`、`saga_model_config.json` 等）  
> - **模型假设**：除非特别说明，以下对比都以 **完整配置 `saga_model_config.json`（大 DiT：embed_dim=1536, depth=24, num_heads=24）** 为准，而不是 small json。

---

## 1. 论文信息

- **论文标题**  
  `SAGA-SR: Semantically and Acoustically Guided Audio Super-Resolution`

- **arXiv 条目**  
  - 链接（工具检索）：<https://arxiv.org/abs/2509.24924>

- **本地论文内容**  
  - 仓库中的 `SAGA_SR_repro.md` 已经包含了论文的主体内容（摘要、方法、实验等），
    其中第 1–3 节就是论文核心描述。

---

## 2. 差异总览（高优先级）

> 下表只列出对训练/推理行为影响最大的高层差异，具体细节在后文模块化展开。

| 方面 | 论文设定 | 当前实现（完整 json） | 差异说明 |
| --- | --- | --- | --- |
| 训练数据规模与域 | 多个音乐/音效/语音库，总计约 3800 小时，通用音频 SR | 主要在 VCTK 语音（甚至其子集）上训练/调试 | 数据规模与域分布差异巨大，论文模型先验更强、更通用 |
| 训练片段长度与裁剪 | 44.1kHz、随机截取 5.94 s 片段 | 44.1kHz、固定使用 1.48 s (`num_samples=65536`)，且始终从开头裁剪 | 片段短且无随机起点，内容多样性与上下文都更弱 |
| 声道处理 | 使用 2 声道表示 `xh, xl ∈ R^{2×L}` | 先将多声道平均成 mono，再复制成 2 声道 | VAE/DiT 实际看到的是“伪立体声”而非真实左右声道分布 |
| Flow Matching 训练 | 使用 DiT + conditional flow matching，按公式 (1)(2)(3) 训练；配置上用 RF objective | config 中 `diffusion.diffusion_objective` 未设置，默认仍为 `"v"`；你在 `training_step` 手写了简化版 Flow Matching | 数学目标接近，但未使用 Stable Audio 官方 RF wrapper 及其时间采样/shift 细节 |
| 预训练权重 | 论文仅说明“基于 DiT backbone”，未说明用 Stable Audio 生成模型权重 | 从 Stable Audio 生成模型(`stable-audio-open-1.0/model.safetensors`)载入权重，并手动扩展部分层以适配新的输入维度 | 你是在文本生成模型上微调为 SR+Flow Matching，这在论文中未出现 |
| roll-off 条件（训练 vs 推理） | roll-off 是核心条件：训练和推理都用 fh, fl | 训练创建数据集时将 `compute_rolloff=False` → 训练阶段完全不算 roll-off；推理/评估时才算并使用 | **训练时没有 roll-off 条件，而推理时使用 roll-off 条件 → train/test 条件严重不一致** |
| LR latent dropout | 对 `zl` 应用 10% dropout，用于 CFG | 代码中实现了 `_apply_latent_dropout`，但默认 `latent_dropout_prob=0`，如果不显式调参则无 dropout | 若按默认参数跑，LR latent 的 10% dropout 实际没有生效，与论文不同 |
| 文本条件来源 | 始终用 Qwen2-Audio 生成 caption，再用 T5-base 编码；10% text dropout | caption 来源可以是缓存、转写 `.txt`、Qwen2-Audio，甚至可被关闭；text dropout 通过“空字符串”方式实现 | 文本条件来源更混合，且可被禁用，与论文“固定使用 Qwen caption”不完全一致 |

> 注意：下面的章节会按模块展开，每一项都给出：**论文设定 / 当前实现（完整 json） / 差异说明 / 相关代码位置**。

---

## 3. 数据与预处理

### 3.1 训练数据与任务范围

| 维度 | 论文设定 | 当前实现 | 差异说明 | 代码位置 |
| --- | --- | --- | --- | --- |
| 任务范围 | 通用音频 SR：音乐、语音、音效 | 目前主要只用 VCTK（语音） | 模型学到的是“窄域语音 SR”，而论文是“通用音频 SR”，先验能力不同 | 数据目录与 CLI 参数（如 `--train_dir`） |
| 数据规模 | ≈3800 小时，多个公开数据集 | 显著小得多（VCTK 语音 + 你选的子集） | 缩小数据规模会让模型表达能力上限大幅下降，尤其是高频与语义先验 | `SAGA_SR_repro.md` §3.1 + 本地数据 |

### 3.2 训练片段长度与裁剪策略

| 维度 | 论文设定 | 当前实现 | 差异说明 | 代码位置 |
| --- | --- | --- | --- | --- |
| 片段长度 | 5.94 s (`L ≈ 5.94 * 44100`) | `duration=1.48`，`num_samples=65536` | 训练片最多为论文约四分之一长度，语境更短 | `train_saga_sr.py` 中构造 `SAGASRDataset` 时的参数 |
| 裁剪位置 | 从整段音频中**随机**截取 | 固定从起点截取：`start = 0` | 缺少“随机起点”会降低内容多样性，模型更易过拟合特定开头模式 | `dataset.py::__getitem__` 中的裁剪逻辑 |

### 3.3 声道处理

| 维度 | 论文设定 | 当前实现 | 差异说明 | 代码位置 |
| --- | --- | --- | --- | --- |
| 声道 | `xh, xl ∈ R^{2×L}`，2 声道 | 先平均到 mono，再 `repeat` 成 2 声道 | 实际输入给 VAE/DiT 的是“左右完全相同”的声道，与真实立体声分布不同 | `dataset.py` 中对 `hr_audio` 的处理逻辑 |

---

## 4. VAE 与 latent 表示

> 本节假设使用完整 `saga_model_config.json` 中的 VAE 配置。

| 维度 | 论文设定 | 当前实现 | 差异说明 | 代码位置 |
| --- | --- | --- | --- | --- |
| VAE 结构 | 使用 Stable Audio 提出的 VAE，[12]，`latent_dim=64, downsampling=2048` | 使用 `stable-audio-open-1.0/vae_model_config.json` 创建的 VAE，`latent_dim=64, downsampling=2048, io_channels=2` | 结构和超参数与论文对齐 | `stable-audio-open-1.0/vae_model_config.json` |
| 训练片段大小 | 隐含对应 5.94 s | 顶层 `sample_size=65536`（1.48 s） | 这是为适配当前训练片段长度的修改，不是 VAE 结构差异；但与论文训练片段长短不同 | `saga_model_config.json` 顶层 `sample_size` |

---

## 5. DiT 配置与 Flow Matching 训练（完整 JSON）

### 5.1 Diffusion objective 与训练 wrapper

| 维度 | 论文设定 | 当前实现（完整 json） | 差异说明 | 代码位置 |
| --- | --- | --- | --- | --- |
| 训练目标 | conditional flow matching，使用路径 (1) 和速度 (2) | config 中 `diffusion.diffusion_objective` 未设置 → `create_diffusion_cond_from_config` 默认 `"v"`；你在 `training_step` 中手写 flow matching | 数学目标大致等价，但你未使用 Stable Audio 官方 `DiffusionCondTrainingWrapper` 中的 `"rectified_flow"` 分支及其时间抽样/shift 逻辑 | `SAGA_SR_repro.md` §2.1；`stable_audio_tools/training/diffusion.py`；`train_saga_sr.py::training_step` |
| 路径定义 | `z_t = (1−t)·z0 + t·z1`，`z1 = zh`（数据）、`z0 ~ N(0,1)`（噪声） | `z_t = (1−t)·hr_latent + t·noise`（t=0 数据、t=1 噪声），`noise ~ N(0,1)` | 本质是把“数据/噪声”的角色互换 + 时间方向反转，属于等价重参数化，但实现细节不同 | `train_saga_sr.py::training_step` |
| 速度向量 | `v_t = z1 − z0` | `v_target = noise − hr_latent` | 速度向量与论文公式方向一致（噪声减数据），与上面路径一起看是一个反向路径 | 同上 |
| timestep 抽样 | t ~ Uniform[0,1] 或文中未细说，但 RF 官方实现支持多种采样/shift | 你是 `t = torch.rand(batch_size)`，没有使用 `dist_shift` 或 log-snr 等策略 | 少了一些稳定训练的细节调参空间 | `train_saga_sr.py::training_step`；`stable_audio_tools/training/diffusion.py` |

### 5.2 DiT 结构（完整 JSON）

| 维度 | 论文设定 | 当前实现（完整 json） | 差异说明 | 代码位置 |
| --- | --- | --- | --- | --- |
| DiT 类型 | 采用 [12] 的 DiT 结构 | `"type": "dit"`，`transformer_type": "continuous_transformer"` | 与论文引用一致 | `saga_model_config.json` 中 `model.diffusion` 部分 |
| 通道维 | latent 通道 64，`zh, zl ∈ R^{64×L/2048}` | `io_channels=64`, `input_concat_dim=64` | 对齐 | 同上 |
| 模型尺寸 | 大模型：embed_dim=1536, depth=24, num_heads=24 | 完整 json 中正是这组参数 | 与论文大模型完全对应 | 同上 |
| 条件 token 维度 | 文本 T5-base 输出 768 维 | `cond_token_dim=768`，`conditioning.cond_dim=768` | 对齐 | `saga_model_config.json` |
| prepended cond dim | 用于 roll-off + 时间的 prepend embedding | `prepend_cond_dim=1536` | 设计对齐，只是你目前训练中没实际用 roll-off | `saga_model_config.json` + `conditioner_rolloff.py` |

### 5.3 预训练权重

| 维度 | 论文设定 | 当前实现 | 差异说明 | 代码位置 |
| --- | --- | --- | --- | --- |
| DiT 初始化 | 只说明“采用 [12] 的架构并用 flow matching 训练”，未明确使用 Stable Audio 文本生成模型权重 | 从 `stable-audio-open-1.0/model.safetensors` 载入 Stable Audio diffusion 权重；并手动扩展 `preprocess_conv.weight` 与 `transformer.project_in.weight` 以适配通道数变更 | 你是从“文本生成音频的 v-objective 模型”出发微调为 SR+Flow Matching，这在论文中没有出现，任务先验不完全匹配 | `train_saga_sr.py` 中加载 stable-audio 权重的函数；`stable-audio-open-1.0/model_config.json` |

---

## 6. 条件：文本、roll-off、LR latent（完整 JSON）

### 6.1 文本条件（caption + T5）

| 维度 | 论文设定 | 当前实现 | 差异说明 | 代码位置 |
| --- | --- | --- | --- | --- |
| caption 来源 | 训练：HR 音频 → Qwen2-Audio；推理：LR 音频 → Qwen2-Audio | `_resolve_caption` 顺序：缓存 → `.txt` 转写 → Qwen2-Audio → 空字符串；还可以用 CLI 完全关闭 caption | 文本条件来源混合，不一定总是 Qwen2-Audio 生成 caption，甚至可以不存在；与论文“固定 Qwen caption”不同 | `audio_captioning_adapter.py`；`train_saga_sr.py::_resolve_caption`；命令行参数 `--disable_caption` / `--use_caption` |
| 文本编码 | T5-base，768 维 token，通过 cross-attention 提供给 DiT | `conditioning` 里 `type: "t5"`，`t5_model_name: "t5-base"`，`cond_dim=768`；DiT 的 `cond_token_dim=768` | 与论文对齐 | `saga_model_config.json` |
| 文本 dropout | 对文本 embedding 使用 10% dropout（论文 §2.2） | 代码中，通过在 10% 概率下把 caption 置为空字符串实现 text dropout（训练时） | dropout 思路对齐，只是实现方式从 embedding 级别变成 caption 级别 | `train_saga_sr.py` 中 `# 论文标准: 10% text dropout` 段落 |

### 6.2 roll-off 条件（**关键差异**）

| 维度 | 论文设定 | 当前实现 | 差异说明 | 代码位置 |
| --- | --- | --- | --- | --- |
| roll-off 计算 | 从 HR/LR 的 STFT（`n_fft=2048, hop=512, Hann`）中计算 0.985 roll-off；沿时间维求和，得到单一标量；用 min–max 归一化到 [0,1)，再 Fourier embedding | `spectral_features.py` 和 `conditioner_rolloff.py` 实现了完全相同的流程（STFT 参数、rolloff 百分比、Fourier embedding 一致）；归一化用固定 [0, 22050] 频率范围，而非数据集级 min–max | 算法形式和参数是对齐的，但归一化用物理频率范围替代了“数据集 min–max”；这一点相比“训练根本没用 roll-off”属于次要差异 | `spectral_features.py::compute_spectral_rolloff/normalize_rolloff`；`conditioner_rolloff.py::_normalize_rolloff`；`SAGA_SR_repro.md` §2.2 |
| roll-off 在训练中的使用 | 训练和推理都使用 roll-off（fh, fl），是 DiT 的两大关键条件之一 | 训练数据集创建时：`SAGASRDataset(..., compute_rolloff=False, ...)` → `metadata` 中没有 `rolloff_low/high`；训练步骤中 `if 'rolloff_low' in metadata[0]` 条件始终为假 → rolloff_cond 恒为 `{'cross_attn': None, 'global': None}`；但推理/评估时会重新计算 roll-off 并使用 | **训练阶段完全没有 roll-off 条件，而推理阶段按论文方式使用 roll-off 条件，这是目前实现中最严重的 train/test 条件不一致，也是和论文差异最大的地方之一。** | `train_saga_sr.py` 中构造 `SAGASRDataset` 的参数；`training_step` 中 rolloff_cond 分支；`inference_saga_sr.py` 与 `evaluate_saga_sr.py` 中的 rolloff 计算 |
| roll-off 作为控制量 | 用户可在推理时调整 target normalized roll-off scalar 控制高频能量（论文 Fig.3） | 你的 `RolloffFourierConditioner` 和采样代码同样允许传入指定 target roll-off（理论上可以实现相同控制），但目前主要是按 LR/HR 实测 roll-off 自动设定 | 控制能力潜力存在，但需要显式接口和实验才能达到论文展示的可控效果 | `conditioner_rolloff.py`；`inference_saga_sr.py` |

### 6.3 LR latent 条件与 dropout

| 维度 | 论文设定 | 当前实现 | 差异说明 | 代码位置 |
| --- | --- | --- | --- | --- |
| LR latent 作用 | `zl` 与 `z_t` 在 channel 维 concat；对 `zl` 用 10% dropout，支持 CFG（条件/无条件） | 训练时：`lr_latent_cond = _apply_latent_dropout(lr_latent, training=True)`，但默认 `latent_dropout_prob=0`；推理时同样直接用 `lr_latent`，未用 10% dropout | 如果按默认参数跑，**LR latent 的 10% dropout 实际没有生效**，与论文描述不同；需要在 config 或 CLI 中显式设置为 0.1 | `train_saga_sr.py::_apply_latent_dropout` 与 trainer 初始化参数 |

---

## 7. 采样与 CFG（完整 JSON）

### 7.1 采样器和时间步调度

| 维度 | 论文设定 | 当前实现 | 差异说明 | 代码位置 |
| --- | --- | --- | --- | --- |
| 采样器 | Euler ODE solver | 使用 `sample_discrete_euler`（Stable Audio 官方实现） | 对齐 | `saga_sampling.py::sample_cfg_euler`；`stable_audio_tools/inference/sampling.py` |
| t-schedule | 线性–二次混合（linear-quadratic），100 步 | 使用 `build_linear_quadratic_t_schedule(num_steps, emulate_linear_steps=250, sigma_max=1.0)`；`num_steps` 由推理脚本参数控制 | 形式对齐；步数可以与论文设为 100 一致 | `t_schedule.py`；`saga_sampling.py` |
| 使用 EMA 模型 | 通常使用 EMA 权重做评估/采样（Stable Audio 推荐实践，但论文未详细描述） | 当前采样直接用在线权重（未见单独的 EMA 模型用于推理） | 少了 EMA 可能让推理结果更噪、稳定性略差，这是一种实现层面的减配 | `train_saga_sr.py` 的保存/加载逻辑 |

### 7.2 多条件 CFG

| 维度 | 论文设定 | 当前实现 | 差异说明 | 代码位置 |
| --- | --- | --- | --- | --- |
| CFG 公式 | 式 (4)：`u_CFG = u_uncond + sa(u_acoustic - u_uncond) + st(u_full - u_acoustic)` | `SAGASRCFGWrapper` 精确实现该线性组合，分别计算 uncond / acoustic / full 三个输出并按公式组合 | CFG 数学形式与论文一致 | `saga_sampling.py::SAGASRCFGWrapper` |
| 条件内容 | `uncond`: 只有 rolloff；`acoustic`: + `zl`；`full`: + `zl` + text | 你的 wrapper 中：  
  - `v_uncond`：`input_concat_cond=0`，不传 text，仅有 rolloff；  
  - `v_acoustic`：`input_concat_cond=lr_latent`，不传 text；  
  - `v_full`：`input_concat_cond=lr_latent`，传入 text+rolloff； | 与论文设定对齐；差异在于训练阶段是否实际使用了这些条件（尤其 rolloff 和 text） | 同上 |

---

## 8. 推理与评估

### 8.1 推理端 caption 与 roll-off 使用

| 维度 | 论文设定 | 当前实现 | 差异说明 | 代码位置 |
| --- | --- | --- | --- | --- |
| 推理 caption | 始终从 LR 音频用 Qwen2-Audio 生成 caption | 只有在 `use_caption=True` 且本地/远程 Qwen 模型可用时才生成；否则可能使用外部传入 caption 或为空 | 推理端语义条件强度和来源可变，比论文更灵活但也更不稳定；若不用 Qwen caption，则和论文设置不一致 | `inference_saga_sr.py`；`audio_captioning_adapter.py` |
| 推理 roll-off | 从 LR/HR 计算 fh, fl，用作 acoustic condition，并可手动调节 target roll-off 标量控制高频能量 | 推理和评估脚本中同样重算 rolloff，并通过 `RolloffFourierConditioner` 提供给 DiT | 推理端 rolloff 使用方式与论文对齐；问题主要出在训练阶段未使用 rolloff | `inference_saga_sr.py`；`evaluate_saga_sr.py` |

### 8.2 评估设置

| 维度 | 论文设定 | 当前实现 | 差异说明 | 代码位置 |
| --- | --- | --- | --- | --- |
| 评估数据集 | 语音：VCTK；音乐：FMA-small；音效：ESC50 fold-5，各 400 条 | 评估脚本主要针对某一数据集目录（例如 VCTK），不含音乐/音效默认配置 | 评估范围更窄，无法复现“通用音频 SR”的整套实验 | `SAGA_SR_repro.md` §3.3；`evaluate_saga_sr.py` |
| 评估指标 | LSD（全部），OpenL3-FD（音乐/音效），主观听测 | 代码实现了 LSD、SI-SDR、MSE、OpenL3-FD，但实际运行时一般只用其中部分指标 | 指标集合能力是具备的，但默认评估配置与论文不完全一致 | `metrics.py`；`evaluate_saga_sr.py` |

---

## 9. 小结（面向完整 JSON 模型）

- **论文名称**：`SAGA-SR: Semantically and Acoustically Guided Audio Super-Resolution`  
- **本对比默认假设你在使用完整的 `saga_model_config.json`（embed_dim=1536, depth=24, num_heads=24 的大 DiT），而不是 small json**。

在此假设下，与论文的关键结构/设定差异主要有：

1. **训练阶段未启用 roll-off 条件，而推理/评估阶段在使用 roll-off 条件**；
2. **LR latent 未按默认 10% dropout 参与 CFG（需要显式设置 `latent_dropout_prob=0.1` 才对齐）**；
3. **Flow Matching 训练未使用 Stable Audio 官方 RF wrapper（`diffusion_objective="rectified_flow"` + `DiffusionCondTrainingWrapper`），而是手写简化版本**；
4. **DiT 使用了 Stable Audio 文本生成模型的预训练权重，并通过手工 patch 扩展维度；论文未提及这种初始化方式**；
5. **数据规模/数据域、训练片段长度和裁剪策略与论文不同：3800 小时通用音频 vs 小规模 VCTK 语音、5.94 s 随机片段 vs 1.48 s 固定首段**；
6. **文本条件的来源与是否启用不固定（缓存/转写/Qwen/禁用），而论文始终使用 Qwen2-Audio caption + T5_base**；
7. **采样端未显式使用 EMA 模型，仅用在线权重进行采样**。

这些差异里，**第 1、2、3 点属于“架构/目标层面”的硬差异**，对模型是否能学到论文那种行为影响最大；
第 4–7 点则是“训练初始化、数据与工程细节”上的差异，会影响最终质量和稳定性。
