#!/usr/bin/env python
"""批量生成 SAGA-SR 数据集字幕缓存的实用脚本。

用法示例（默认路径为 `/data01/audio_group/m24_yuanjiajun/mdct_sr/data/wav48`）::

    python generate_captions.py --base-dir /data01/audio_group/m24_yuanjiajun/mdct_sr/data/wav48 \
        --output caption_cache.pt

脚本会依次遍历 `train/`、`val/`、`infer/` 子目录，调用 `pregenerate_captions`
生成或追加 `caption_cache.pt`，确保训练/验证/推理阶段都能直接
使用缓存的文本描述而无需再次加载 Qwen2-Audio 模型。

注意：
  * 同一份缓存文件可安全存放训练集和验证集的字幕，因为两者来自
    同一数据源拆分，音频文件路径各不相同，不会互相覆盖。
  * 如果已经存在缓存条目，`pregenerate_captions` 会自动跳过，支持断点续跑。
  * 若只想处理其中部分子集，可以通过 `--include` 参数指定。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from audio_captioning_adapter import pregenerate_captions


DEFAULT_BASE_DIR = Path("/data01/audio_group/m24_yuanjiajun/mixed_vctk_hifitts")
DEFAULT_OUTPUT = Path("cache.pt")
SUBSETS = {
    "train": True,   # 训练阶段使用高分辨率音频描述
    "eval": True,     # 验证集同样使用高分辨率描述
    "test": False,  # 推理阶段可使用低分辨率描述
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量生成 SAGA-SR 数据集的音频字幕缓存",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="包含 train/ val/ infer/ 子目录的根目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="字幕缓存输出路径 (默认: caption_cache.pt)",
    )
    parser.add_argument(
        "--mode",
        choices=("local", "api"),
        default="local",
        help="字幕生成方式：本地 Qwen2-Audio 或 API 模式",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        choices=tuple(SUBSETS.keys()),
        help="仅处理指定子集（默认三个子集都会处理）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="强制启用断点续跑（若输出文件已存在，不会清空）",
    )
    return parser.parse_args()


def ensure_subsets(include: Iterable[str] | None) -> List[str]:
    if include:
        return [name for name in include if name in SUBSETS]
    return list(SUBSETS.keys())


def main() -> None:
    args = parse_args()
    subsets = ensure_subsets(args.include)

    base_dir = args.base_dir
    output_cache = args.output

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    print(f"Base directory: {base_dir}")
    print(f"Output cache : {output_cache}")
    print(f"Mode         : {args.mode}")
    print(f"Subsets      : {', '.join(subsets)}")
    print("=== 开始批量生成字幕 ===")

    for subset in subsets:
        subset_dir = base_dir / subset
        use_hr_audio = SUBSETS[subset]

        if not subset_dir.exists():
            print(f"[跳过] {subset} 子目录不存在: {subset_dir}")
            continue

        print(f"\n>>> 处理子集: {subset} ({'HR' if use_hr_audio else 'LR'})")
        pregenerate_captions(
            audio_dir=str(subset_dir),
            output_cache=str(output_cache),
            mode=args.mode,
            use_hr_audio=use_hr_audio,
        )

    print("\n全部子集处理完成。缓存文件可直接用于训练/验证/推理。")


if __name__ == "__main__":
    main()
