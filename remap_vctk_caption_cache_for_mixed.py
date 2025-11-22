#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""将旧 VCTK caption_cache 中的路径批量 remap 到 mixed_vctk_hifitts 目录。

典型场景：
- 旧缓存中的 key：
    /data01/audio_group/m24_yuanjiajun/mdct_sr/data/wav48/{split}/p227/p227_001.wav
- 新 mixed 数据集中的路径：
    /data01/audio_group/m24_yuanjiajun/mixed_vctk_hifitts/{split}/vctk/p227/p227_001.wav

本脚本会：
- 读取 caption_cache.pt
- 对匹配到旧 VCTK 根目录的 key，根据 split=train/eval/test 映射到 mixed 根目录
- 将新的 key 写回同一个 cache（旧 key 保留），并在覆盖前自动创建 .bak 备份

用法示例（在远程服务器 speechsr 项目根目录下）：

    conda activate speechsr  # 或你的环境
    cd /data01/audio_group/m24_yuanjiajun/speechsr

    # 先 dry-run 看看映射统计，不写回
    python remap_vctk_caption_cache_for_mixed.py --dry_run

    # 确认无误后正式写回（会自动生成 caption_cache.pt.bak* 备份）
    python remap_vctk_caption_cache_for_mixed.py

注意：
- 建议在训练停止时运行，避免和训练过程同时读写 caption_cache.pt。
"""

import argparse
import os
from typing import Dict, Tuple

import torch


def remap_vctk_keys(
    cache: Dict[str, str],
    old_root: str,
    new_root: str,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """根据给定 old_root/new_root，将 VCTK 路径 remap 到 mixed 根目录。

    映射规则：
    - old_root/{split}/<rel_path> -> new_root/{split}/vctk/<rel_path>
      其中 split ∈ {train, eval, test}

    返回：
    - new_cache: 带有新 key 的完整缓存（在原 cache 基础上追加）
    - stats: 一些统计信息
    """
    splits = ["train", "eval", "test"]

    # 标准化根目录，去掉末尾的斜杠
    old_root = old_root.rstrip("/\\")
    new_root = new_root.rstrip("/\\")

    new_cache: Dict[str, str] = dict(cache)

    total_keys = len(cache)
    matched = 0
    added = 0
    skipped_existing = 0

    for key, caption in cache.items():
        # 只处理字符串 key
        if not isinstance(key, str):
            continue

        for split in splits:
            old_prefix = f"{old_root}/{split}/"
            if key.startswith(old_prefix):
                matched += 1
                rel_path = key[len(old_prefix) :]
                new_key = f"{new_root}/{split}/vctk/{rel_path}"

                if new_key in new_cache:
                    # 已存在（可能是之前在线训练时为新路径生成过 caption）
                    skipped_existing += 1
                else:
                    new_cache[new_key] = caption
                    added += 1
                break

    stats = {
        "total_keys": total_keys,
        "matched_old_vctk": matched,
        "new_keys_added": added,
        "existing_new_keys_skipped": skipped_existing,
    }
    return new_cache, stats


def make_backup_path(path: str) -> str:
    """生成不覆盖已有文件的备份路径。

    若 path.bak 不存在，则使用它；否则依次尝试 path.bak1, path.bak2, ...
    """
    base = path + ".bak"
    if not os.path.exists(base):
        return base

    idx = 1
    while True:
        candidate = f"{base}{idx}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap VCTK paths in caption_cache.pt to mixed_vctk_hifitts root",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="caption_cache.pt",
        help="Path to caption cache file (default: caption_cache.pt)",
    )
    parser.add_argument(
        "--old_root",
        type=str,
        default="/data01/audio_group/m24_yuanjiajun/mdct_sr/data/wav48",
        help="Old VCTK root directory used when generating original captions",
    )
    parser.add_argument(
        "--new_root",
        type=str,
        default="/data01/audio_group/m24_yuanjiajun/mixed_vctk_hifitts",
        help="New mixed dataset root directory containing vctk subfolder",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not write back cache file, only print statistics",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not create backup .bak file before overwriting cache",
    )

    args = parser.parse_args()

    if not os.path.exists(args.cache_path):
        raise FileNotFoundError(
            f"Cache file not found: {args.cache_path}. "
            f"请确认你在和训练脚本同一个目录下运行，或者显式指定 --cache_path"
        )

    print(f"Loading cache from: {args.cache_path}")
    cache = torch.load(args.cache_path, map_location="cpu")

    if not isinstance(cache, dict):
        raise TypeError(
            f"Expected cache to be a dict, got {type(cache)}. "
            f"请确认这是 CaptionCache 保存的文件。"
        )

    print(f"Loaded {len(cache)} entries from cache")
    print(f"Old VCTK root: {args.old_root}")
    print(f"New mixed root: {args.new_root}")

    new_cache, stats = remap_vctk_keys(cache, args.old_root, args.new_root)

    print("\n=== Remap statistics ===")
    print(f"Total keys in original cache     : {stats['total_keys']}")
    print(f"Matched old VCTK keys            : {stats['matched_old_vctk']}")
    print(f"New mixed VCTK keys added        : {stats['new_keys_added']}")
    print(f"Existing mixed keys not overwritten: {stats['existing_new_keys_skipped']}")
    print("================================\n")

    if args.dry_run:
        print("Dry-run mode: not writing back cache file.")
        return

    # 写回前备份原文件
    if not args.no_backup:
        backup_path = make_backup_path(args.cache_path)
        print(f"Creating backup: {backup_path}")
        torch.save(cache, backup_path)

    print(f"Writing updated cache back to: {args.cache_path}")
    torch.save(new_cache, args.cache_path)
    print("Done.")


if __name__ == "__main__":
    main()
