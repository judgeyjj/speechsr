# 条件 (Conditioning)
在 `stable-audio-tools` 的上下文中，条件（Conditioning）是指在模型中使用额外的信号，以便对模型的行为增加一层额外的控制。例如，我们可以根据文本提示来限定扩散模型的输出，从而创建一个文本转音频模型。

# 条件类型
根据所使用的条件信号，有几种不同类型的条件。

## 交叉注意力 (Cross attention)
交叉注意力是一种条件类型，它允许我们找到两个可能不同长度的序列之间的相关性。例如，交叉注意力可以帮助我们找到文本编码器的特征序列和高层音频特征序列之间的相关性。

用于交叉注意力条件的信号形状应为 `[batch, sequence, channels]`。

## 全局条件 (Global conditioning)
全局条件是使用单个 n 维张量来提供与整个被条件序列相关的条件信息。例如，这可以是 CLAP 模型的单个嵌入输出，或一个学习到的类别嵌入。

用于全局条件的信号形状应为 `[batch, channels]`。

## 前置条件 (Prepend conditioning)
前置条件涉及将条件 tokens 预置到模型中的数据 tokens 之前，从而允许信息通过模型的自注意力机制进行解释。

目前，这种条件类型仅受基于 Transformer 的模型（如扩散 Transformer）支持。

用于前置条件的信号形状应为 `[batch, sequence, channels]`。

## 输入拼接 (Input concatenation)
输入拼接将一个空间条件信号应用于模型，该信号在序列维度上与模型的输入相关，并且长度相同。条件信号将沿着通道维度与模型的输入数据拼接在一起。这可以用于像修复（inpainting）信息、旋律条件，或创建扩散自编码器等任务。

用于输入拼接条件的信号形状应为 `[batch, channels, sequence]`，并且必须与模型的输入长度相同。

# 条件器和条件配置
`stable-audio-tools` 使用条件器（Conditioner）模块将人类可读的元数据（如文本提示或秒数）转换为模型可以接受的张量输入。

每个条件器都有一个对应的 `id`，它期望在训练或推理期间提供的条件字典中找到这个 `id`。每个条件器接收相关的条件数据，并返回一个包含相应张量和掩码的元组。

`ConditionedDiffusionModelWrapper` 负责管理用户提供的元数据字典（例如 `{"prompt": "一首优美的歌曲", "seconds_start": 22, "seconds_total": 193}`）和模型使用的不同条件类型字典（例如 `{"cross_attn_cond": ...}`）之间的转换。

要对模型应用条件，你必须在模型的配置中提供一个 `conditioning` 配置。目前，我们仅通过 `diffusion_cond` 模型类型支持对扩散模型进行条件化。

`conditioning` 配置应包含一个 `configs` 数组，允许你定义多个条件信号。

`configs` 数组中的每一项都应定义相应元数据的 `id`、要使用的条件器类型以及该条件器的配置。

`cond_dim` 属性用于强制所有条件输入具有相同的维度，但这可以通过在任何单个配置上显式设置 `output_dim` 属性来覆盖。

## 配置示例
```json
"conditioning": {
    "configs": [
        {
            "id": "prompt",
            "type": "t5",
            "config": {
                "t5_model_name": "t5-base",
                "max_length": 77,
                "project_out": true
            }
        }
    ],
    "cond_dim": 768
}
```

# 条件器 (Conditioners)

## 文本编码器

### `t5`
它使用来自 `transformers` 库的一个冻结的 [T5](https://huggingface.co/docs/transformers/model_doc/t5) 文本编码器，将文本提示编码为一个文本特征序列。

`t5_model_name` 属性决定从 `transformers` 库加载哪个 T5 模型。

`max_length` 属性决定了文本编码器将接受的最大 token 数量，以及输出文本特征的序列长度。

如果将 `enable_grad` 设置为 `true`，T5 模型将被解冻并随模型检查点一起保存，从而允许你对 T5 模型进行微调。

T5 编码仅与交叉注意力条件兼容。

#### 配置示例
```json
{
    "id": "prompt",
    "type": "t5",
    "config": {
        "t5_model_name": "t5-base",
        "max_length": 77,
        "project_out": true
    }
}
```

### `clap_text`
它从 [CLAP](https://github.com/LAION-AI/CLAP) 模型中加载文本编码器，可以提供文本特征序列或单个多模态文本/音频嵌入。

必须为 CLAP 模型提供一个本地文件路径，在 `clap_ckpt_path` 属性中设置，同时还要为所提供的模型提供正确的 `audio_model_type` 和 `enable_fusion` 属性。

如果 `use_text_features` 属性设置为 `true`，条件器输出将是文本特征序列，而不是单个多模态嵌入。这允许模型使用更细粒度的文本信息，但代价是失去了使用 CLAP 音频嵌入进行提示的能力。

默认情况下，如果 `use_text_features` 为 true，则返回 CLAP 文本编码器特征的最后一层。你可以通过在 `feature_layer_ix` 属性中指定要返回的层的索引来返回早期层的文本特征。例如，通过将 `feature_layer_ix` 设置为 `-2`，可以返回 CLAP 模型倒数第二层的文本特征。

如果将 `enable_grad` 设置为 `true`，CLAP 模型将被解冻并随模型检查点一起保存，从而允许你对 CLAP 模型进行微调。

CLAP 文本嵌入与全局条件和交叉注意力条件兼容。如果 `use_text_features` 设置为 `true`，则特征与全局条件不兼容。

#### 配置示例
```json
{
    "id": "prompt",
    "type": "clap_text",
    "config": {
        "clap_ckpt_path": "/path/to/clap/model.ckpt",
        "audio_model_type": "HTSAT-base",
        "enable_fusion": true,
        "use_text_features": true,
        "feature_layer_ix": -2
    }
}
```

## 数字编码器

### `int`
`IntConditioner` 接收给定范围内的整数列表，并为每个整数返回一个离散的学习到的嵌入。

`min_val` 和 `max_val` 属性设置嵌入值的范围。输入的整数将被限制在此范围内。

这可以用于离散的时间嵌入或学习到的类别嵌入等。

整数嵌入与全局条件和交叉注意力条件兼容。

#### 配置示例
```json
{
    "id": "seconds_start",
    "type": "int",
    "config": {
        "min_val": 0,
        "max_val": 512
    }
}
```

### `number`
`NumberConditioner` 接收给定范围内的浮点数列表，并返回所提供浮点数的连续傅里叶嵌入。

`min_val` 和 `max_val` 属性设置浮点值的范围。此范围用于归一化输入的浮点值。

数字嵌入与全局条件和交叉注意力条件兼容。

#### 配置示例
```json
{
    "id": "seconds_total",
    "type": "number",
    "config": {
        "min_val": 0,
        "max_val": 512
    }
}
```