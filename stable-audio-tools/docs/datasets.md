# 数据集
`stable-audio-tools` 支持从本地文件存储加载数据，以及从 Amazon S3 存储桶加载 [WebDataset](https://github.com/webdataset/webdataset/tree/main/webdataset) 格式的音频文件和 JSON 文件。

# 数据集配置
要指定用于训练的数据集，你必须向 `train.py` 提供一个数据集配置 JSON 文件。

数据集配置包括一个 `dataset_type` 属性，用于指定要使用的数据加载器类型；一个 `datasets` 数组，用于提供多个数据源；以及一个 `random_crop` 属性，该属性决定从训练样本中裁剪的音频是从音频文件中的随机位置还是始终从开头开始。

## 本地音频文件
要使用本地的音频样本目录，请在你的数据集配置中将 `dataset_type` 属性设置为 `"audio_dir"`，并向 `datasets` 属性提供一个对象列表，其中包含 `path` 属性，该属性应为你的音频样本目录的路径。

这将从提供的目录及其所有子目录中加载所有兼容的音频文件。

### 配置示例
```json
{
    "dataset_type": "audio_dir",
    "datasets": [
        {
            "id": "my_audio",
            "path": "/path/to/audio/dataset/"
        }
    ],
    "random_crop": true
}
```

## S3 WebDataset
要从托管在 Amazon S3 存储桶中的 WebDataset 格式的 .tar 文件加载音频文件和相关元数据，你可以将 `dataset_type` 属性设置为 `s3`，并向 `datasets` 参数提供一个对象列表，其中包含指向 WebDataset .tar 文件的共享 S3 存储桶前缀的 AWS S3 路径。S3 存储桶将根据给定路径进行递归搜索，并假定找到的任何 .tar 文件都包含音频文件和相应的 JSON 文件，其中相关文件仅在文件扩展名上有所不同（例如 "000001.flac", "000001.json", "00002.flac", "00002.json" 等）。

### 配置示例
```json
{
    "dataset_type": "s3",
    "datasets": [
        {
            "id": "s3-test",
            "s3_path": "s3://my-bucket/datasets/webdataset/audio/"
        }
    ],
    "random_crop": true
}
```

## 预编码数据集
要使用通过[预编码脚本](pre_encoding.md)创建的预编码潜在变量，请将 `dataset_type` 属性设置为 `"pre_encoded"`，并提供包含预编码的 `.npy` 潜在文件和相应 `.json` 元数据文件的目录路径。

你可以选择性地指定一个 `latent_crop_length`（以潜在单位表示，其中潜在长度 = `audio_samples // 2048`），以将预编码的潜在变量裁剪到比你编码时更小的长度。如果未指定，则使用完整的预编码长度。当 `random_crop` 设置为 true 时，它将根据你期望的 `latent_crop_length` 从序列中随机裁剪，同时考虑填充。

**注意**：`random_crop` 目前不会更新 `seconds_start`，因此在用于训练或微调使用该条件的模型（例如 `stable-audio-open-1.0`）时，该值将不准确，但可以用于不使用 `seconds_start` 的模型（例如 `stable-audio-open-small`）。

### 配置示例
```json
{
    "dataset_type": "pre_encoded",
    "datasets": [
        {
            "id": "my_pre_encoded_audio",
            "path": "/path/to/pre_encoded/output/",
            "latent_crop_length": 512,
            "custom_metadata_module": "/path/to/custom_metadata.py"
        }
    ],
    "random_crop": true
}
```

有关创建预编码数据集的信息，请参见[预编码](pre_encoding.md)。

# 自定义元数据
要在模型训练期间自定义提供给条件器的元数据，你可以向数据集配置提供一个单独的自定义元数据模块。该元数据模块应为一个 Python 文件，其中必须包含一个名为 `get_custom_metadata` 的函数，该函数接收两个参数 `info` 和 `audio`，并返回一个字典。

对于本地训练，`info` 参数将包含一些关于已加载音频文件的信息，例如路径以及音频是如何从原始训练样本中裁剪的。对于 WebDataset 数据集，它还将包含来自相关 JSON 文件的元数据。

`audio` 参数包含将在训练时传递给模型的音频样本。这使你可以分析音频以获取额外的属性，然后将这些属性作为额外的条件信号传入。

从 `get_custom_metadata` 函数返回的字典的属性将被添加到训练时使用的 `metadata` 对象中。有关条件如何工作的更多信息，请参阅[条件文档](./conditioning.md)。

## 配置和自定义元数据模块示例
```json
{
    "dataset_type": "audio_dir",
    "datasets": [
        {
            "id": "my_audio",
            "path": "/path/to/audio/dataset/",
            "custom_metadata_module": "/path/to/custom_metadata.py",
        }
    ],
    "random_crop": true
}
```

`custom_metadata.py`:
```py
def get_custom_metadata(info, audio):

    # 将音频文件的相对路径作为提示传入
    return {"prompt": info["relpath"]}
```