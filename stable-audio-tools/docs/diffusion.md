# 扩散 (Diffusion)

扩散模型学习对数据进行去噪。

# 模型配置
扩散模型的模型配置文件应将 `model_type` 设置为 `diffusion_cond`（如果模型使用条件）或 `diffusion_uncond`（如果不使用），并且 `model` 对象应具有以下属性：

- `diffusion`
    - 扩散模型本身的配置。有关扩散模型配置的更多信息，请参见下文。
- `pretransform`
    - 扩散模型的[预转换](pretransforms.md)配置，例如用于潜在扩散的自编码器。
    - 可选
- `conditioning`
    - 扩散模型的各种[条件](conditioning.md)模块的配置。
    - 仅 `diffusion_cond` 需要
- `io_channels`
    - 扩散模型的基本输入/输出通道数。
    - 推理脚本使用此参数来确定为扩散模型生成的噪声的形状。

# 扩散配置
- `type`
    - Transformer 的底层模型类型。
    - 对于有条件的扩散模型，应为 `dit` ([扩散 Transformer](#diffusion-transformers-dit))、`DAU1d` ([Dance Diffusion U-Net](#dance-diffusion-u-net)) 或 `adp_cfg_1d` ([audio-diffusion-pytorch U-Net](#audio-diffusion-pytorch-u-net-adp)) 之一。
    - 无条件的扩散模型也可以使用 `adp_1d`。
- `cross_attention_cond_ids`
    - 用作交叉注意力输入的条件信息的条件器 ID。
    - 如果指定了多个 ID，条件张量将沿着序列维度拼接。
- `global_cond_ids`
    - 用作全局条件输入的条件信息的条件器 ID。
    - 如果指定了多个 ID，条件张量将沿着通道维度拼接。
- `prepend_cond_ids`
    - 预置到模型输入之前的条件信息的条件器 ID。
    - 如果指定了多个 ID，条件张量将沿着序列维度拼接。
    - 仅适用于扩散 Transformer 模型。
- `input_concat_ids`
    - 拼接到模型输入的条件信息的条件器 ID。
    - 如果指定了多个 ID，条件张量将沿着通道维度拼接。
    - 如果条件张量的长度与模型输入不同，它们将沿着序列维度进行插值以使其长度相同。
        - 插值算法依赖于具体模型，但通常使用最近邻重采样。
- `config`
    - 模型骨干网络本身的配置。
    - 依赖于具体模型。

# 训练配置
扩散模型配置文件中的 `training` 配置应具有以下属性：

- `learning_rate`
    - 训练期间使用的学习率。
    - 默认为恒定学习率，可以用 `optimizer_configs` 覆盖。
- `use_ema`
    - 如果为 true，则在训练期间会维护一个模型权重的副本，并作为训练模型权重的指数移动平均值进行更新。
    - 可选。默认值：`true`
- `log_loss_info`
    - 如果为 true，将在所有 GPU 上收集额外的扩散损失信息，并在训练期间显示。
    - 可选。默认值：`false`
- `loss_configs`
    - 损失函数计算的配置。
    - 可选
- `optimizer_configs`
    - 优化器和调度器的配置。
    - 可选，会覆盖 `learning_rate`。
- `demo`
    - 训练期间演示的配置，包括条件信息。
- `pre_encoded`
    - 如果为 true，表示模型应在[预编码的潜在变量](pre_encoding.md)上操作，而不是原始音频。
    - 在使用[预编码数据集](datasets.md#pre-encoded-datasets)进行训练时是必需的。
    - 可选。默认值：`false`

## 配置示例
```json
"training": {
    "use_ema": true,
    "log_loss_info": false,
    "optimizer_configs": {
        "diffusion": {
            "optimizer": {
                "type": "AdamW",
                "config": {
                    "lr": 5e-5,
                    "betas": [0.9, 0.999],
                    "weight_decay": 1e-3
                }
            },
            "scheduler": {
                "type": "InverseLR",
                "config": {
                    "inv_gamma": 1000000,
                    "power": 0.5,
                    "warmup": 0.99
                }
            }
        }
    },
    "demo": { ... }
}
```

# 演示配置
扩散模型训练配置中的 `demo` 配置应具有以下属性：
- `demo_every`
    - 两次演示之间的训练步数。
- `demo_steps`
    - 为演示运行的扩散时间步数。
- `num_demos`
    - 每个演示中要生成的样本数量。
- `demo_cond`
    - 对于有条件的扩散模型，这是提供给每个样本的条件元数据，以列表形式提供。
    - 注意：列表长度必须与 `num_demos` 相同。
- `demo_cfg_scales`
    - 对于有条件的扩散模型，这提供了一个在演示期间渲染的无分类器指导 (CFG) 尺度的列表。这有助于了解随着训练的进行，模型对不同条件强度的响应。

## 配置示例
```json
"demo": {
    "demo_every": 2000,
    "demo_steps": 250,
    "num_demos": 4,
    "demo_cond": [
        {"prompt": "A beautiful piano arpeggio", "seconds_start": 0, "seconds_total": 80},
        {"prompt": "A tropical house track with upbeat melodies, a driving bassline, and cheery vibes", "seconds_start": 0, "seconds_total": 250},
        {"prompt": "A cool 80s glam rock song with driving drums and distorted guitars", "seconds_start": 0, "seconds_total": 180},
        {"prompt": "A grand orchestral arrangement", "seconds_start": 0, "seconds_total": 190}
    ],
    "demo_cfg_scales": [3, 6, 9]
}
```

# 模型类型

多种不同的模型类型可以用作扩散模型的底层骨干网络。目前，这包括 U-Net 和 Transformer 模型的变体。

## 扩散 Transformers (DiT)

在模型质量方面，Transformer 通常持续优于 U-Net，但它们对内存和计算的要求更高，并且最适合在较短的序列上工作，例如音频的潜在编码。

### Continuous Transformer

这是我们自定义的 Transformer 模型实现，基于 `x-transformers` 的实现，但进行了效率改进，例如融合的 QKV 层和 Flash Attention 2 支持。

### `x-transformers`

此模型类型使用来自 https://github.com/lucidrains/x-transformers 仓库的 `ContinuousTransformerWrapper` 类作为扩散 Transformer 骨干网络。

`x-transformers` 是一个很好的基准 Transformer 实现，为各种实验性设置提供了许多选项。
它非常适合在不自己实现的情况下测试实验性功能，但实现可能没有完全优化，并且可能会在没有太多警告的情况下引入破坏性更改。

## 扩散 U-Net

U-Net 使用分层架构在进行更繁重的处理之前逐步下采样输入数据，然后再次上采样数据，使用跳跃连接将数据跨越下采样“谷”（名称中的“U”）传递到相同分辨率的上采样层。

### audio-diffusion-pytorch U-Net (ADP)

此模型类型使用了来自 `https://github.com/archinetai/audio-diffusion-pytorch` 仓库 0.0.94 版本的 `UNetCFG1D` 类的修改实现，并增加了 Flash Attention 支持。

### Dance Diffusion U-Net

这是 [Dance Diffusion](https://github.com/Harmonai-org/sample-generator) 中使用的 U-Net 的重新实现。它的条件支持非常有限，基本上只支持全局条件。主要用于无条件的扩散模型。