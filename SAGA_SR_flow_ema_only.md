# SAGA-SR 关键差异（仅 Flow Matching & EMA）

> 前提假设：  
> - 使用完整 `saga_model_config.json`（大 DiT：embed_dim=1536, depth=24, num_heads=24）；  
> - 始终使用 Qwen2-Audio 文本条件；  
> - 忽略你为 small / debug 显式改动的部分（数据规模、roll-off 开关等）。  
> 
> 本文档**只保留两个差异**：  
> 1. Flow Matching 训练目标（差异 3）  
> 2. EMA 使用（差异 7）  

---

## 1. Flow Matching 训练目标（差异 3）

### 1.1 论文 / Stable 官方 / 当前实现

- **论文 SAGA-SR（`SAGA_SR_repro.md` §2.1）**  
  使用 conditional flow matching：
  - 路径：
    \[
    z_t = (1-t)\cdot z_0 + t\cdot z_1,\quad z_1 = z_h,\ z_0\sim\mathcal N(0,I)
    \]
  - 速度：
    \[
    v_t = z_1 - z_0
    \]
  - 损失：
    \[
    \mathbb E\|u(z_t, z_l, c, f_h, f_l, t;\theta) - v_t\|^2
    \]
  本质是：**噪声↔数据之间的一条线性路径 + 常向量速度场**。

- **Stable Audio 官方 `rectified_flow`（`DiffusionCondTrainingWrapper`）**  
  当 `diffusion_objective="rectified_flow"` 时：
  ```python
  # data = diffusion_input, noise ~ N(0, I)
  alphas, sigmas = 1 - t, t
  noised_inputs = diffusion_input * alphas + noise * sigmas
  # 等价于 z_t = (1 - t) * data + t * noise

  targets = noise - diffusion_input  # 噪声 - 数据
  ```
  若记 `z_1 = data`, `z_0 = noise`，则：
  - 路径：`z_t = (1-t) z_1 + t z_0`  
  - 速度：`v_t = z_0 - z_1`  

  和论文只是把 `z0/z1` 命名对调、时间反向，本质仍是**同一条噪声–数据直线路径的 Flow Matching**。

- **你当前实现（`SAGASRTrainer.training_step`）**  
  ```python
  noise = torch.randn_like(hr_latent)
  z_t = (1 - t[:, None, None]) * hr_latent + t[:, None, None] * noise
  v_target = noise - hr_latent
  v_pred = self.model.model(z_t, t, **conditioning_inputs)
  loss = F.mse_loss(v_pred, v_target)
  ```
  - 同样是“线性插值 + 噪声减数据”，与官方 RF 数学上非常接近；
  - 但：
    - config 里没有显式 `diffusion_objective="rectified_flow"`；
    - 没有用官方 `DiffusionCondTrainingWrapper`；
    - 没有 `timestep_sampler` / `dist_shift` 等成熟的时间调度实现。

### 1.2 对齐选择：论文还是 Stable 官方？

- **理论层面**：论文定义了 Flow Matching 的数学目标（直线路径 + 常向量速度）。
- **工程层面**：Stable Audio 已经在相同类型的 DiT 上实现并验证了 `rectified_flow` 训练。

二者在目标函数上是**数学等价**的，只是符号和时间参数化方式不同。  
因此更合理的策略是：

> **在数学目标上对齐论文，在具体代码实现上对齐 Stable Audio 官方 `rectified_flow`。**

### 1.3 具体对齐方案（Flow Matching）

**目标：把 Flow Matching 部分改成『论文目标 + Stable 官方实现』。**

1. **配置：显式设置 RF 目标**  
   在完整 `saga_model_config.json` 的 `diffusion` 段增加：
   ```json
   "diffusion": {
     "type": "dit",
     "...": "...",
     "diffusion_objective": "rectified_flow"
   }
   ```

2. **训练入口：改用官方 `DiffusionCondTrainingWrapper`**  
   - 使用 `stable_audio_tools.training.factory.create_training_wrapper_from_config(model_config, model)`：
     - 自动选择 `DiffusionCondTrainingWrapper`；
     - Wrapper 负责：t 抽样、构造 `noised_inputs`、`targets`、调用 `self.diffusion`；
   - 你需要保证：
     - VAE pretransform 正确（已经有）；
     - conditioner 正确（文本、roll-off、LR latent、CFG dropout）；
     - training config 中的 `learning_rate`、`timestep_sampler` 等符合论文设定（比如 `uniform` / `log_snr`）。

3. **采样端：继续使用你现有的 CFG + Euler 实现**  
   - `SAGASRCFGWrapper + sample_cfg_euler` 数学上已经和论文 CFG 公式 (4) 对齐；
   - 只要训练端切换到官方 RF，实现路径就完全是“论文目标 + Stable 官方代码”。

---

## 2. EMA 使用（差异 7）

### 2.1 论文 / Stable 官方 / 当前实现

- **论文 SAGA-SR**  
  - 论文正文没有详细描述 EMA，但在扩散模型实践中，几乎都是：
    - 训练：用 online 权重更新；
    - 推理/导出：用 EMA 权重。

- **Stable Audio 官方做法**  
  在 `DiffusionCondTrainingWrapper` 中：
  ```python
  from ema_pytorch import EMA

  if use_ema:
      self.diffusion_ema = EMA(
          self.diffusion.model,
          beta=0.9999,
          power=3/4,
          update_every=1,
          update_after_step=1,
          include_online_model=False,
      )
  else:
      self.diffusion_ema = None
  ```

  - 每个 step 在 `on_before_zero_grad` 中：
    ```python
    if self.diffusion_ema is not None:
        self.diffusion_ema.update()
    ```
  - demo / 采样 / 导出 checkpoint 时：
    ```python
    model = module.diffusion_ema.ema_model if module.diffusion_ema is not None else module.diffusion.model
    ```
  - 即：**训练用 online 模型，推理/导出优先用 EMA 模型**。

- **你当前实现**  
  - 自己写的 `SAGASRTrainer` 中，没有任何 EMA 相关逻辑；
  - 推理脚本和采样直接用 `self.model.model` 的 online 权重。

### 2.2 对齐选择：论文还是 Stable 官方？

- 论文没有给出具体 EMA 实现；
- Stable Audio 提供了完整的、已经在相同 DiT 结构上验证过的 EMA 方案；

因此：

> **在 EMA 上，以 Stable Audio 官方实现为对齐目标。**

### 2.3 具体对齐方案（EMA）

这里给出两条路线，你可以分阶段采用。

#### 路线 A：随 Flow Matching 一起，直接使用官方 `DiffusionCondTrainingWrapper`（推荐）

1. 在 training config 中打开 EMA：
   ```json
   "training": {
     "learning_rate": 1e-5,
     "batch_size": 4,
     "use_ema": true,
     "...": "..."
   }
   ```

2. 通过 `create_training_wrapper_from_config` 创建 `DiffusionCondTrainingWrapper` 时，会自动：
   - 构建 `diffusion_ema`；
   - 在 `on_before_zero_grad` 中更新 EMA；
   - 在 demo / 导出模型时使用 `ema_model`；

3. 推理脚本中加载的 checkpoint，即为 EMA 权重，对应 Stable 官方使用方式。

#### 路线 B：在现有 `SAGASRTrainer` 上手工加入轻量 EMA

如果短期内不想迁移到官方 wrapper，可以在你自己的 trainer 上手动加一个简易 EMA：

1. 在 `SAGASRTrainer.__init__` 中增加：
   ```python
   from ema_pytorch import EMA

   self.model_ema = EMA(
       self.model.model,
       beta=0.9999,
       power=3/4,
       update_every=1,
       update_after_step=1,
   )
   ```

2. 在训练循环中（例如 `on_before_zero_grad`）调用：
   ```python
   def on_before_zero_grad(self, *args, **kwargs):
       if hasattr(self, "model_ema") and self.model_ema is not None:
           self.model_ema.update()
   ```

3. 推理时优先使用 EMA 模型：
   - 保存 checkpoint 时另存一份 `ema_state_dict` 或专门导出 `self.model_ema.ema_model`；
   - 推理脚本加载这份 EMA 权重，用于采样。

这条路线改动较小，可以验证“online vs EMA 推理”的质量差异，为后续完全迁移到官方 wrapper 提供经验。

---

## 3. 总结：对齐策略一览

- **Flow Matching（差异 3）**：
  - 数学目标：对齐 SAGA-SR 论文 (1)(2)(3) 的 Flow Matching（直线噪声–数据路径 + 常向量速度）。
  - 代码实现：对齐 Stable Audio 官方 `rectified_flow`（`diffusion_objective="rectified_flow"`），在训练/验证中直接使用官方 `(alphas, sigmas)` 与 `targets` 公式，而不是继续手写版本。

- **EMA（差异 7）**：
  - 论文未给细节；
  - 代码实现：对齐 Stable Audio 官方 EMA 方案，在训练中维护 EMA，在导出推理权重时优先使用 EMA 模型。

---

## 4. 本仓库当前已经实现的改动（Flow + EMA）

这一节描述 `speechsr/` 里 **已经落地** 的 Flow Matching 与 EMA 对齐改动，并标出哪些属于 *SAGA 论文未写、为对齐 Stable Audio 官方实现而做的工程选择*。

### 4.1 配置层对齐

- 文件：`saga_model_config.json` / `saga_model_config_small.json`

- **`diffusion.diffusion_objective = "rectified_flow"`**
  - 作用：令 DiTWrapper 与上层训练逻辑以 Rectified Flow 目标工作。
  - 论文关系：
    - 论文给出了 Flow Matching 的数学目标，但**没有出现 `diffusion_objective` 这个字段名**；
    - 该键名来自 Stable Audio，实现上与论文目标等价。

- **`training.use_ema = true`、`training.cfg_dropout_prob = 0.1`、`training.timestep_sampler = "uniform"`**
  - 作用：
    - `use_ema`：打开 EMA；
    - `cfg_dropout_prob`：设置 CFG dropout 概率（10%）；
    - `timestep_sampler`：指定时间步采样策略（目前为均匀）。
  - 论文关系：
    - 论文只在概念上提到“10% 文本 dropout”“t ~ U(0,1)`，**没有给出这些具体字段名**；
    - 这些键完全沿用 Stable Audio 官方配置，是*工程层*的对齐。

### 4.2 训练端 Flow Matching 对齐

- 文件：`train_saga_sr.py`

- **`training_step` / `validation_step (val_use_flowmatch)` 重写**
  - 改动：
    - 顶部新增 `get_alphas_sigmas` 导入：
      - `from stable_audio_tools.inference.sampling import sample_discrete_euler, get_alphas_sigmas`
    - 在 `training_step` / FlowMatch 验证分支中：
      - 统一读取 `diffusion_objective = getattr(self.model, "diffusion_objective", "rectified_flow")`；
      - 若目标为 `"v"`：调用 **Stable Audio 官方** `get_alphas_sigmas(t)`；
      - 若目标为 `"rectified_flow"/"rf_denoiser"`：使用官方 `(alphas, sigmas) = (1-t, t)`；
      - 按官方公式组合 `z_t = data * alphas + noise * sigmas`；
      - 目标：
        - `v` 目标：`targets = noise * alphas - data * sigmas`；
        - `rectified_flow` 目标：`targets = noise - data`。
  - 论文关系：
    - 论文只给出路径与速度的数学形式，**没有给出 `get_alphas_sigmas` 这种函数名和实现细节**；
    - 这里完全采用 Stable Audio 的实现，是*在数学上与论文等价、在代码层面对齐官方*。

### 4.3 EMA 接入与更新策略

- 文件：`train_saga_sr.py`

- **初始化 EMA（`__init__`）**
  - 逻辑：
    - 读取 `training_cfg = self.config.get("training", {})`；
    - 若 `training_cfg.get("use_ema", True)`：
      - 创建 `self.model_ema = EMA(self.model.model, beta=0.9999, power=3/4, update_every=1, update_after_step=1, include_online_model=False)`；
    - 否则 `self.model_ema = None`。
  - 与官方关系：
    - 参数 `(beta=0.9999, power=3/4, update_every=1, update_after_step=1, include_online_model=False)` 与 Stable Audio `DiffusionCondTrainingWrapper` 完全一致；
    - **这些超参数在 SAGA 论文中没有给出，纯属 Stable Audio 官方工程细节，我们在此逐项对齐。**

- **更新时机（`on_before_zero_grad`）**
  - 新增：
    - `def on_before_zero_grad(self, *args, **kwargs):` 中：
      - 若 `self.model_ema` 存在，则 `self.model_ema.update()`。
  - 与官方关系：
    - 与 `DiffusionCondTrainingWrapper.on_before_zero_grad` 的用法一致；
    - 更新时机属于实现细节，论文未说明，我们直接对齐官方。

### 4.4 EMA 导出与推理使用

- **导出函数：`SAGASRTrainer.export_ema_model`**
  - 行为：
    - 备份当前 `self.model.model`；
    - 若存在 EMA：
      - 将 `self.model.model` 临时替换为 `self.model_ema.ema_model`；
      - 否则保持 online 模型；
    - 保存：`{"state_dict": self.model.state_dict()}` 到指定路径；
    - 最后恢复 `self.model.model`。
  - 与官方关系：
    - 模式与 `DiffusionCondTrainingWrapper.export_model` 相同，只是多了一层 `SAGASRTrainer` 包装；
    - **导出文件名本身（例如 `saga_sr_final_ema.ckpt`）是工程约定，论文中未出现。**

- **训练脚本末尾导出 EMA checkpoint**
  - 在 `main()` 中：
    - 仍保存 Lightning 自身 checkpoint：`saga_sr_final.ckpt`（含优化器等）；
    - 新增：
      - `ema_path = os.path.join(args.output_dir, 'saga_sr_final_ema.ckpt')`
      - `model.export_ema_model(ema_path)`
  - 使用建议：
    - 推理脚本 `inference_saga_sr.py` 调用时，推荐：
      - `--checkpoint path/to/saga_sr_final_ema.ckpt`；
    - 这样加载到的就是 EMA 平滑后的权重，与 Stable Audio 官方推荐用法一致。

### 4.5 小结：哪些改动是 SAGA 论文里没写的？

综合上面几小节，**以下内容均属于论文未写，但为对齐 Stable Audio 官方实现而在本仓库中加入的工程选择**：

- `model.diffusion.diffusion_objective` 字段本身及其取值字符串（例如 `"rectified_flow"`、`"v"`）；
- `training.use_ema`、`training.cfg_dropout_prob`、`training.timestep_sampler` 等具体配置键；
- `(alphas, sigmas)` 的具体函数形式（`get_alphas_sigmas` 与 `(1-t, t)`），以及在代码中如何组合 `z_t` 与 `targets`；
- EMA 的全部超参数：`beta=0.9999`、`power=3/4`、`update_every=1`、`update_after_step=1`、`include_online_model=False`；
- EMA 的导出与文件命名约定，例如 `saga_sr_final_ema.ckpt` 及其在推理脚本中的使用方式。

这些改动都遵循同一原则：

> **在数学目标层面严格对齐 SAGA-SR 论文，在实现细节层面尽量完全复用 Stable Audio 官方 rectified_flow + EMA 代码路径。**

