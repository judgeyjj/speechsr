# 自编码器
在高层次上，自编码器是由两部分组成的模型：一个*编码器* (encoder) 和一个*解码器* (decoder)。

*编码器*接收一个序列（例如单声道或立体声音频），并输出该序列的压缩表示，形式为一个 d 通道的“潜在序列”，通常会以一个固定的系数进行大幅度的下采样。

*解码器*接收一个 d 通道的潜在序列，并将其上采样回原始输入序列的长度，从而逆转编码器的压缩过程。

自编码器通过结合重建损失和对抗性损失进行训练，目的是为原始音频数据创建一个紧凑且可逆的表示。这种表示允许下游模型在一个数据压缩的“潜在空间”中工作，该空间具有各种理想且可控的属性，例如序列长度减小、抗噪声能力和离散化。

在 `stable-audio-tools` 中定义的自编码器架构主要是全卷积的，这使得在短序列上训练的自编码器可以应用于任意长度的序列。例如，一个在1秒样本上训练的自编码器，可以用来将45秒的输入编码为潜在扩散模型所需的潜在表示。

# 模型配置
自编码器的模型配置文件应将 `model_type` 设置为 `autoencoder`，并且 `model` 对象应具有以下属性：

- `encoder`
    - 自编码器编码器部分的配置
- `decoder`
    - 自编码器解码器部分的配置
- `latent_dim`
    - 自编码器的潜在维度，供推理脚本和下游模型使用
- `downsampling_ratio`
    - 输入序列与潜在序列之间的下采样比率，供推理脚本和下游模型使用
- `io_channels`
    - 当输入和输出通道数相同时，指定自编码器的通道数，供推理脚本和下游模型使用
- `bottleneck`
    - 自编码器瓶颈部分的配置
    - 可选
- `pretransform`
    - 为自编码器定义一个预转换，例如小波分解或另一个自编码器
    - 更多信息请参见 [pretransforms.md](pretransforms.md)
    - 可选
- `in_channels`
    - 当输入通道数与 `io_channels` 不同时（例如在单声道转立体声模型中），指定自编码器的输入通道数
    - 可选
- `out_channels`
    - 当输出通道数与 `io_channels` 不同时，指定自编码器的输出通道数
    - 可选

# 训练配置
自编码器模型配置文件中的 `training` 配置应具有以下属性：
- `learning_rate`
    - 训练期间使用的学习率
- `use_ema`
    - 如果为 true，则在训练期间会维护一个模型权重的副本，并作为训练模型权重的指数移动平均值进行更新。
    - 可选。默认值：`false`
- `warmup_steps`
    - 在启用对抗性损失之前的训练步数
    - 可选。默认值：`0`
- `encoder_freeze_on_warmup`
    - 如果为 true，则在预热步骤完成后冻结编码器，因此对抗性训练只影响解码器。
    - 可选。默认值：`false`
- `loss_configs`
    - 损失函数计算的配置
    - 可选
- `optimizer_configs`
    - 优化器和调度器的配置
    - 可选

## 损失配置
用于自编码器训练的损失有几种不同类型，包括谱损失、时域损失、对抗性损失和瓶颈特定的损失。

这些损失的超参数以及损失权重因子可以在 `training` 配置的 `loss_configs` 属性中进行配置。

### 谱损失
多分辨率短时傅里叶变换（STFT）损失是我们音频自编码器使用的主要重建损失。我们使用 [auraloss](https://github.com/csteinmetz1/auraloss/tree/main/auraloss) 库来实现我们的谱损失函数。

对于单声道自编码器 (`io_channels` == 1)，我们使用 [MultiResolutionSTFTLoss](https://github.com/csteinmetz1/auraloss/blob/1576b0cd6e927abc002b23cf3bfc455b660f663c/auraloss/freq.py#L329) 模块。

对于立体声自编码器 (`io_channels` == 2)，我们使用 [SumAndDifferenceSTFTLoss](https://github.com/csteinmetz1/auraloss/blob/1576b0cd6e927abc002b23cf3bfc455b660f663c/auraloss/freq.py#L533) 模块。

#### 配置示例
```json
"spectral": {
    "type": "mrstft",
    "config": {
        "fft_sizes": [2048, 1024, 512, 256, 128, 64, 32],
        "hop_sizes": [512, 256, 128, 64, 32, 16, 8],
        "win_lengths": [2048, 1024, 512, 256, 128, 64, 32],
        "perceptual_weighting": true
    },
    "weights": {
        "mrstft": 1.0
    }
}
```

### 时域损失
我们计算原始音频和解码后音频之间的 L1 距离，以提供一个时域损失。

#### 配置示例
```json
"time": {
    "type": "l1",
    "weights": {
        "l1": 0.1
    }
}
```

### 对抗性损失
对抗性损失引入一个鉴别器模型集成，用于区分真实音频和生成音频，从而为自编码器提供关于需要修复的感知差异的信号。

我们主要依赖于 EnCodec 仓库中的[多尺度 STFT 鉴别器](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/msstftd.py#L99)。

#### 配置示例
```json
"discriminator": {
    "type": "encodec",
    "config": {
        "filters": 32,
        "n_ffts": [2048, 1024, 512, 256, 128],
        "hop_lengths": [512, 256, 128, 64, 32],
        "win_lengths": [2048, 1024, 512, 256, 128]
    },
    "weights": {
        "adversarial": 0.1,
        "feature_matching": 5.0
    }
}
```

## 演示配置
对于自编码器训练的演示，唯一需要设置的属性是 `demo_every`，它决定了两次演示之间的步数。

### 配置示例
```json
"demo": {
    "demo_every": 2000
}
```

# 编码器和解码器类型
编码器和解码器在模型配置中是分开定义的，因此来自不同模型架构和库的编码器和解码器可以互换使用。

## Oobleck
Oobleck 是 Harmonai 内部的自编码器架构，它实现了来自多种其他自编码器架构的特性。

### 配置示例
```json
"encoder": {
    "type": "oobleck",
    "config": {
        "in_channels": 2,
        "channels": 128,
        "c_mults": [1, 2, 4, 8],
        "strides": [2, 4, 8, 8],
        "latent_dim": 128,
        "use_snake": true
    }
},
"decoder": {
    "type": "oobleck",
    "config": {
        "out_channels": 2,
        "channels": 128,
        "c_mults": [1, 2, 4, 8],
        "strides": [2, 4, 8, 8],
        "latent_dim": 64,
        "use_snake": true,
        "use_nearest_upsample": false
    }
}
```

## DAC
这是来自 `descript-audio-codec` 仓库的编码器和解码器定义。它是一个简单的全卷积自编码器，每一层的通道数都会翻倍。编码器和解码器的配置会直接传递给 DAC [Encoder](https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/model/dac.py#L64) 和 [Decoder](https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/model/dac.py#L115) 的构造函数。

**注意：这不包括 DAC 的量化器，也不会加载预训练的 DAC 模型，这只是编码器和解码器的定义。**

### 配置示例
```json
"encoder": {
    "type": "dac",
    "config": {
        "in_channels": 2,
        "latent_dim": 32,
        "d_model": 128,
        "strides": [2, 4, 4, 4]
    }
},
"decoder": {
    "type": "dac",
    "config": {
        "out_channels": 2,
        "latent_dim": 32,
        "channels": 1536,
        "rates": [4, 4, 4, 2]
    }
}
```

## SEANet
这是来自 Meta 的 EnCodec 仓库的 SEANetEncoder 和 SEANetDecoder 定义。这与 MusicGen 中使用的 EnCodec 模型中的编码器和解码器架构相同，但不包含量化器。

编码器和解码器的配置会直接传递给 [SEANetEncoder](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/modules/seanet.py#L66C12-L66C12) 和 [SEANetDecoder](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/modules/seanet.py#L147) 类，不过我们颠倒了编码器中步幅（ratios）的输入顺序，以使其与解码器中的顺序保持一致。

### 配置示例
```json
"encoder": {
    "type": "seanet",
    "config": {
        "channels": 2,
        "dimension": 128,
        "n_filters": 64,
        "ratios": [4, 4, 8, 8],
        "n_residual_layers": 1,
        "dilation_base": 2,
        "lstm": 2,
        "norm": "weight_norm"
    }
},
"decoder": {
    "type": "seanet",
    "config": {
        "channels": 2,
        "dimension": 64,
        "n_filters": 64,
        "ratios": [4, 4, 8, 8],
        "n_residual_layers": 1,
        "dilation_base": 2,
        "lstm": 2,
        "norm": "weight_norm"
    }
},
```

# 瓶颈 (Bottlenecks)
在我们的术语中，自编码器的“瓶颈”是一个放置在编码器和解码器之间的模块，用于对编码器创建的潜在空间施加特定的约束。

瓶颈具有与自编码器类似的接口，定义了 `encode()` 和 `decode()` 函数。一些瓶颈除了输出潜在序列外，还会返回额外的信息，例如量化后的 token 索引，或在训练期间需要考虑的额外损失。

要为自编码器定义一个瓶颈，你可以在自编码器的模型配置中提供 `bottleneck` 对象。

## VAE

变分自编码器 (Variational Autoencoder, VAE) 瓶颈将编码器的输出沿通道维度一分为二，将这两半分别视为 VAE 采样的“均值”和“尺度”参数，并执行潜在采样。在基本层面上，“尺度”值决定了要添加到“均值”潜在变量中的噪声量，这创建了一个抗噪声的潜在空间，使得潜在空间中更大部分的区域都能解码为感知上“有效”的音频。这对于扩散模型尤其有帮助，因为扩散采样过程的输出会残留一些高斯误差噪声。

**注意：要使 VAE 瓶颈正常工作，编码器的输出维度必须是解码器输入维度的两倍。**

### 配置示例
```json
"bottleneck": {
    "type": "vae"
}
```

### 额外信息
VAE 瓶颈还会在编码器信息中返回一个 `kl` 值。这是编码/采样后的潜在空间与高斯分布之间的 [KL 散度](https://zh.wikipedia.org/wiki/K-L%E6%95%A3%E5%BA%A6)。通过将此值作为优化的损失值，我们能将我们的潜在分布推向更接近正态分布，但这可能会以牺牲重建质量为代价。

### 损失配置示例
```json
"bottleneck": {
    "type": "kl",
    "weights": {
        "kl": 1e-4
    }
}
```

## Tanh
这个瓶颈将 `tanh` 函数应用于潜在序列，将潜在值“软裁剪”到 -1 和 1 之间。这是一种快速而粗略的方式来强制限制潜在空间的方差，但训练这些模型可能不稳定，因为潜在空间似乎很容易饱和到 -1 或 1 的值并且无法恢复。

### 配置示例
```json
"bottleneck": {
    "type": "tanh"
}
```

## Wasserstein
Wasserstein 瓶颈实现了来自论文 [Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558) 的 WAE-MMD 正则化方法，计算了潜在空间与高斯分布之间的最大均值差异 (Maximum Mean Discrepancy, MMD)。将此值作为优化的损失值，可以使潜在空间更接近高斯分布，但它不需要像 VAE 那样进行随机采样，因此编码器是确定性的。

Wasserstein 瓶颈还暴露了 `noise_augment_dim` 属性，它会将 `noise_augment_dim` 个通道的高斯噪声连接到潜在序列，然后再传递给解码器。这为潜在变量增加了一些随机性，有助于对抗性训练，同时保持编码器输出的确定性。

**注意：对于较长的序列长度，MMD 计算对显存 (VRAM) 的消耗非常大，因此最好在具有相当大下采样率的自编码器上，或在短序列长度上训练 Wasserstein 自编码器。在推理时，MMD 计算是禁用的。**

### 配置示例
```json
"bottleneck": {
    "type": "wasserstein"
}
```

### 额外信息
这个瓶颈会在编码器信息中添加 `mmd` 值，代表最大均值差异。

### 损失配置示例
```json
"bottleneck": {
    "type": "mmd",
    "weights": {
        "mmd": 100
    }
}
```

## L2 归一化 (球形自编码器)
L2 归一化瓶颈在通道维度上对潜在变量进行归一化，将潜在变量投影到一个 d 维超球面上。这起到一种潜在空间归一化的作用。


### 配置示例
```json
"bottleneck": {
    "type": "l2_norm"
}
```


## RVQ
残差矢量量化 (Residual vector quantization, RVQ) 目前是学习离散神经音频编解码器（音频的分词器/tokenizer）的领先方法。在矢量量化中，潜在序列中的每个项都会被单独“吸附”到离散“码本”中最近的一个学习到的向量上。该向量在码本中的索引可以作为 token 索引，用于自回归 Transformer 等模型。残差矢量量化通过增加额外的码本来提高普通矢量量化的精度。想更深入地了解 RVQ，请查看 [Scott Hawley 博士的这篇博客文章](https://drscotthawley.github.io/blog/posts/2023-06-12-RVQ.html)。

这个 RVQ 瓶颈使用了来自 `vector-quantize-pytorch` 仓库的 [lucidrains 的实现](https://github.com/lucidrains/vector-quantize-pytorch/tree/master)，它提供了许多不同的量化器选项。瓶颈配置会直接传递给 `ResidualVQ` 的[构造函数](https://github.com/lucidrains/vector-quantize-pytorch/blob/0c6cea24ce68510b607f2c9997e766d9d55c085b/vector_quantize_pytorch/residual_vq.py#L26)。

**注意：这个 RVQ 实现使用手动替换码本向量来减少码本崩溃。这在多 GPU 训练中不起作用，因为随机替换在不同设备之间不是同步的。**

### 配置示例
```json
"bottleneck": {
    "type": "rvq",
    "config": {
        "num_quantizers": 4,
        "codebook_size": 2048,
        "dim": 1024,
        "decay": 0.99,
    }
}
```

## DAC RVQ
这是来自 `descript-audio-codec` 仓库的残差矢量量化实现。它与上述实现的不同之处在于，它不使用手动替换来提高码本使用率，而是使用可学习的线性层在执行单个量化操作之前将潜在变量投影到较低维空间。这意味着它与分布式训练兼容。

瓶颈配置会直接传递给 `ResidualVectorQuantize` 的[构造函数](https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/nn/quantize.py#L97)。

`quantize_on_decode` 属性也被暴露出来，它将量化过程移至解码器。这不应在训练期间使用，但在训练潜在扩散模型时很有用，这些模型使用量化过程作为在扩散采样过程后消除误差的一种方式。

### 配置示例
```json
"bottleneck": {
    "type": "dac_rvq",
    "config": {
        "input_dim": 64,
        "n_codebooks": 9,
        "codebook_dim": 32,
        "codebook_size": 1024,
        "quantizer_dropout": 0.5
    }
}
```

### 额外信息
DAC RVQ 瓶颈还会向 `info` 对象添加以下属性：
- `pre_quantizer`
    - 量化前的潜在序列，与 `quantize_on_decode` 结合使用时，对训练潜在扩散模型很有用。
- `vq/commitment_loss`
    - 量化器的承诺损失
- `vq/codebook_loss`
    - 量化器的码本损失
