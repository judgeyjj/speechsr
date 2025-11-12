# 预编码

在冻结的预训练自编码器上对编码后的潜在变量进行模型训练时，编码器通常是冻结的。因此，通常的做法是预先将音频编码为潜在变量并将其存储在磁盘上，而不是在训练期间即时计算。这样可以提高训练吞吐量，并释放本可用于编码的 GPU 内存。

## 先决条件

要将音频预编码为潜在变量，你需要一个数据集配置文件、一个自编码器模型配置文件和一个**未包装 (unwrapped)** 的自编码器检查点文件。

**注意：**你可以在 `stabilityai/stable-audio-open-1.0` Hugging Face [仓库](https://huggingface.co/stabilityai/stable-audio-open-1.0)中找到未包装的 VAE 检查点（`vae_model.ckpt`）和配置（`vae_config.json`）的副本。这与 `stable-audio-open-small` 中使用的 VAE 相同。

## 运行预编码脚本

要从自编码器模型预编码潜在变量，你可以使用 `pre_encode.py`。该脚本将加载一个预训练的自编码器，对潜在变量/令牌进行编码，并将其以易于在训练期间加载的格式保存到磁盘。

`pre_encode.py` 脚本接受以下命令行参数：

- `--model-config`
  - 模型配置文件的路径
- `--ckpt-path`
  - **未包装 (unwrapped)** 的自编码器模型检查点的路径
- `--model-half`
  - 如果为 true，则对模型权重使用半精度
  - 可选
- `--dataset-config`
  - 数据集配置文件的路径
  - 必选
- `--output-path`
  - 输出文件夹的路径
  - 必选
- `--batch-size`
  - 用于处理的批量大小
  - 可选，默认为 1
- `--sample-size`
  - 用于预编码时填充/裁剪的音频样本数
  - 可选，默认为 1320960 (约 30 秒)
- `--is-discrete`
  - 如果为 true，则将模型视为离散模型，保存离散的令牌而不是连续的潜在变量
  - 可选
- `--num-nodes`
  - 用于分布式处理的节点数（如果可用）。
  - 可选，默认为 1
- `--num-workers`
  - 数据加载器的工作进程数
  - 可选，默认为 4
- `--strategy`
  - PyTorch Lightning 策略
  - 可选，默认为 'auto'
- `--limit-batches`
  - 限制处理的批次数
  - 可选
- `--shuffle`
  - 如果为 true，则打乱数据集
  - 可选

**注意：**在预编码时，建议在你的数据集配置中设置 `"drop_last": false`，以确保即使最后一批不完整也能被处理。

例如，如果你想以半精度编码长度填充至 30 秒的潜在变量，可以运行以下命令：

```bash
$ python3 ./pre_encode.py \
--model-config /path/to/model/config.json \
--ckpt-path /path/to/autoencoder/model.ckpt \
--model-half \
--dataset-config /path/to/dataset/config.json \
--output-path /path/to/output/dir \
--sample-size 1320960 \
```

当你运行上述命令时，`--output-path` 目录将包含用于编码潜在变量的每个 GPU 进程的编号子目录，以及一个 `details.json` 文件，该文件记录了脚本运行时使用的设置。

在编号的子目录内，你会找到编码为 `.npy` 文件的潜在变量，以及相关的 `.json` 元数据文件。

```bash
/path/to/output/dir/
├── 0
│   ├── 0000000000000.json
│   ├── 0000000000000.npy
│   ├── 0000000000001.json
│   ├── 0000000000001.npy
│   ├── 0000000000002.json
│   ├── 0000000000002.npy
...
└── details.json
```

## 在预编码的潜在变量上进行训练

一旦你将潜在变量保存到磁盘，就可以通过向 `train.py` 提供一个指向预编码潜在变量的数据集配置文件来使用它们训练模型，并指定 `"dataset_type"` 为 `"pre_encoded"`。在底层，这将配置一个 `stable_audio_tools.data.dataset.PreEncodedDataset`。有关配置预编码数据集的更多信息，请参阅数据集文档的[预编码数据集](datasets.md#pre-encoded-datasets)部分。

数据集配置文件应如下所示：

```json
{
    "dataset_type": "pre_encoded",
    "datasets": [
        {
            "id": "my_audio",
            "path": "/path/to/output/dir"
        }
    ],
    "random_crop": false
}
```

在你的扩散模型配置中，你还需要在 [`training` 部分](diffusion.md#training-configs)中指定 `pre_encoded: true`，以告知训练包装器在预编码的潜在变量而不是原始音频上操作。

```json
"training": {
    "pre_encoded": true,
    ...
}
```
