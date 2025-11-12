# 预转换 (Pretransforms)
在我们的术语中，“预转换”是一个模块，它在被包装的模型（例如扩散模型）运行之前，对输入数据进行转换。

预转换模块具有 `encode` 和 `decode` 方法。`encode` 方法在训练和推理时，在模型的前向传递之前被调用，而 `decode` 方法在推理时，在模型的前向传递之后被调用，以便将模型的输出转换回原始数据空间（例如，将潜在变量转换回音频）。

## 预转换配置
在模型配置中，预转换模块被定义为一个 `pretransform` 对象。该对象应包含一个 `type` 属性，用于指定要使用的预转换模块的类型，以及一个 `config` 对象，该对象将被传递给预转换模块的构造函数。

### 配置示例
```json
"pretransform": {
    "type": "autoencoder",
    "config": {
        "model_config": "configs/autoencoder/vae_config.json",
        "ckpt_path": "models/autoencoder/vae.ckpt",
        "device": "cuda"
    }
}
```

# 预转换类型

## 自编码器
自编码器预转换模块将一个完整的自编码器模型包装起来，并在训练和推理时使用它。在训练期间，它在将数据传递给下游模型（如扩散模型）之前，使用自编码器的编码器将音频编码为潜在变量。在推理时，它在下游模型运行后，使用自编码器的解码器将潜在变量解码回音频。

自编码器预转换的配置需要一个 `model_config` 属性，该属性指向自编码器的模型配置文件，以及一个 `ckpt_path` 属性，该属性指向自编码器的模型检查点文件。

### 配置示例
```json
"pretransform": {
    "type": "autoencoder",
    "config": {
        "model_config": "configs/autoencoder/vae_config.json",
        "ckpt_path": "models/autoencoder/vae.ckpt",
        "device": "cuda",
        "embed_dim": 4,
        "downsampling_ratio": 512,
        "io_channels": 2,
        "latent_scale": 1
    }
}
```
## 小波变换 (Wavelet)
小波变换预转换模块使用 `pytorch-wavelets` 库中的可逆小波变换，在将数据传递给下游模型之前，将数据分解为多个频带。

`wavelet` 属性设置要使用的小波名称（例如 "db7", "sym2", "bior3.5"）。有关可用小波的完整列表，请参阅 `pywt.wavelist()`。

`levels` 属性设置要执行的小波分解的级别数。每个级别都会将输入数据分解为一个低通频带和多个高通频带。

`mode` 属性设置在执行小波变换时使用的填充模式。

### 配置示例
```json
"pretransform": {
    "type": "wavelet",
    "config": {
        "wavelet": "bior3.5",
        "levels": 5,
        "mode": "reflect"
    }
}
```