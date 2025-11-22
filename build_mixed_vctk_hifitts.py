#!/usr/bin/env python
"""构建 VCTK + HiFiTTS 混合 train/eval/test 数据集。

特点：
- **不修改原始数据目录**：
  - VCTK 根：  /data01/audio_group/m24_yuanjiajun/mdct_sr/data/wav48
  - HiFiTTS 根：/data01/audio_group/m24_yuanjiajun/mdct_sr/hi_fi_tts_v0
- 在一个全新的输出根目录下创建：

    <output_root>/
        train/
            vctk/   # 来自 wav48/train
            hifitts/# 来自 HiFiTTS 官方 train manifest
        eval/
            vctk/   # 来自 wav48/eval
            hifitts/# 来自 HiFiTTS 官方 dev manifest
        test/
            vctk/   # 来自 wav48/test
            hifitts/# 来自 HiFiTTS 官方 test manifest

- VCTK：在新目录中保留与原来相同的子目录结构；
- HiFiTTS：按照官方 manifest 的 train/dev/test 划分放入对应子集；
- 默认使用 **符号链接**（symlink），节省空间；也可选用 copy。

之后可以：

- 在 <output_root> 上运行 generate_captions.py + Qwen2-Audio 重新转录；
- 训练时将 `--train_dir` 指向 `<output_root>/train`，`--val_dir` 指向 `<output_root>/eval`。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple

# 根据你之前提供的路径设定默认值
DEFAULT_VCTK_ROOT = Path("/data01/audio_group/m24_yuanjiajun/mdct_sr/data/wav48")
DEFAULT_HIFITTS_ROOT = Path("/data01/audio_group/m24_yuanjiajun/mdct_sr/hi_fi_tts_v0")
DEFAULT_OUTPUT_ROOT = Path("/data01/audio_group/m24_yuanjiajun/mixed_vctk_hifitts")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建 VCTK + HiFiTTS 的混合 train/eval/test 数据集",
    )
    parser.add_argument(
        "--vctk-root",
        type=Path,
        default=DEFAULT_VCTK_ROOT,
        help="VCTK wav48 根目录，内部应包含 train/ eval/ test 子目录",
    )
    parser.add_argument(
        "--hifitts-root",
        type=Path,
        default=DEFAULT_HIFITTS_ROOT,
        help="HiFiTTS 根目录，内部应包含 audio/ 及若干 manifest JSON",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="混合数据输出根目录，将在其中创建 train/eval/test 子目录",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="symlink",
        help="文件处理方式：copy=物理复制，symlink=符号链接(推荐，节省空间)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划操作，不实际创建/复制/链接文件",
    )
    return parser.parse_args()


def ensure_dir(path: Path, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[DRY] mkdir -p {path}")
        return
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str, dry_run: bool = False) -> None:
    """根据 mode 选择复制或创建符号链接，若目标已存在则跳过。"""
    if dst.exists():
        return
    ensure_dir(dst.parent, dry_run=dry_run)

    if dry_run:
        print(f"[DRY] {mode.upper()} {src} -> {dst}")
        return

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------- VCTK 部分 ---------------- #

def process_vctk(vctk_root: Path, out_root: Path, mode: str, dry_run: bool = False) -> Dict[str, int]:
    """将 VCTK 的 train/eval/test 拷贝/链接到新的输出根目录。

    假设结构：
        vctk_root/
            train/...
            eval/...
            test/...

    输出：
        out_root/{train,eval,test}/vctk/...
    """
    subset_map = {
        "train": "train",
        "eval": "eval",
        "test": "test",
    }

    counts: Dict[str, int] = {"train": 0, "eval": 0, "test": 0}

    for subset, out_subset in subset_map.items():
        src_dir = vctk_root / subset
        if not src_dir.exists():
            print(f"[VCTK] 子集 {subset} 不存在，跳过: {src_dir}")
            continue

        print(f"[VCTK] 处理子集 {subset}: {src_dir}")
        for root, _, files in os.walk(src_dir):
            root_path = Path(root)
            rel_root = root_path.relative_to(src_dir)

            for name in files:
                ext = Path(name).suffix.lower()
                src_path = root_path / name

                if ext in AUDIO_EXTS:
                    dst_path = out_root / out_subset / "vctk" / rel_root / name
                    link_or_copy(src_path, dst_path, mode=mode, dry_run=dry_run)
                    counts[out_subset] += 1
                elif ext == ".txt":
                    # 若 VCTK 本身有 .txt 转录，一并复制/链接过去
                    dst_path = out_root / out_subset / "vctk" / rel_root / name
                    link_or_copy(src_path, dst_path, mode=mode, dry_run=dry_run)

    return counts


# ---------------- HiFiTTS 部分 ---------------- #

def infer_subset_from_manifest(name: str) -> str | None:
    """根据 manifest 文件名推断 train/eval/test 子集。

    约定：文件名以 *_train.json, *_dev.json, *_test.json 结尾：
        *_train.json → train
        *_dev.json   → eval
        *_test.json  → test
    """
    if name.endswith("_train.json"):
        return "train"
    if name.endswith("_dev.json"):
        return "eval"  # dev 当作验证集
    if name.endswith("_test.json"):
        return "test"
    return None


def normalize_hifitts_relpath(audio_filepath: str) -> Path:
    """保留 HiFiTTS 的相对路径，例如 audio/92_clean/.../xxx.flac。"""
    return Path(audio_filepath)


def process_hifitts(hifitts_root: Path, out_root: Path, mode: str, dry_run: bool = False) -> Dict[str, int]:
    """根据 HiFiTTS 官方 manifest 构造 train/eval/test 子集。

    在 hifitts_root 下搜索 `*_manifest_*_{train,dev,test}.json`，每行 JSON:

        {"audio_filepath": "audio/...flac", "text": "...", ...}

    - 源音频： hifitts_root / audio_filepath
    - 目标音频： out_root/<subset>/hifitts/<audio_filepath>
    - 同时在目标处生成同名 `.txt`（写入 text/text_normalized/text_no_preprocessing），以便
      需要时可以直接用 transcript，而不用 Qwen。
    """
    counts: Dict[str, int] = {"train": 0, "eval": 0, "test": 0}

    if not hifitts_root.exists():
        print(f"[HiFiTTS] 根目录不存在: {hifitts_root}")
        return counts

    manifest_files = sorted(hifitts_root.glob("*_manifest_*_*.json"))
    if not manifest_files:
        print(f"[HiFiTTS] 未在 {hifitts_root} 下找到 manifest JSON 文件，检查路径是否正确。")
        return counts

    print(f"[HiFiTTS] 共发现 {len(manifest_files)} 个 manifest 文件。")

    # 避免同一源文件在同一子集被重复处理
    seen: Dict[Tuple[str, Path], bool] = {}

    for manifest in manifest_files:
        subset = infer_subset_from_manifest(manifest.name)
        if subset is None:
            print(f"[HiFiTTS] 跳过无法识别子集的 manifest: {manifest.name}")
            continue

        print(f"[HiFiTTS] 处理 manifest ({subset}): {manifest.name}")

        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"  [WARN] 解析 JSON 行失败: {exc} | 行内容: {line[:80]}...")
                    continue

                audio_rel = entry.get("audio_filepath")
                if not audio_rel:
                    continue

                src_audio = hifitts_root / audio_rel
                if not src_audio.exists():
                    print(f"  [WARN] 音频文件不存在，跳过: {src_audio}")
                    continue

                key = (subset, src_audio.resolve())
                if key in seen:
                    continue
                seen[key] = True

                rel_path = normalize_hifitts_relpath(audio_rel)
                dst_audio = out_root / subset / "hifitts" / rel_path

                # 复制/链接音频
                link_or_copy(src_audio, dst_audio, mode=mode, dry_run=dry_run)
                counts[subset] += 1

                # 写入文本 .txt（可选备用）
                text = (
                    entry.get("text")
                    or entry.get("text_normalized")
                    or entry.get("text_no_preprocessing")
                    or ""
                )
                if text:
                    dst_txt = dst_audio.with_suffix(".txt")
                    if dry_run:
                        print(f"[DRY] write TXT {dst_txt}")
                    else:
                        ensure_dir(dst_txt.parent, dry_run=False)
                        if not dst_txt.exists():
                            with dst_txt.open("w", encoding="utf-8") as tf:
                                tf.write(text.strip())

    return counts


def main() -> None:
    args = parse_args()

    print("=== 构建 VCTK + HiFiTTS 混合数据集 ===")
    print(f"VCTK 根目录   : {args.vctk_root}")
    print(f"HiFiTTS 根目录: {args.hifitts_root}")
    print(f"输出根目录    : {args.output_root}")
    print(f"模式          : {args.mode} ({'仅打印不执行' if args.dry_run else '实际执行'})")

    # 准备输出子目录
    for subset in ("train", "eval", "test"):
        for ds_name in ("vctk", "hifitts"):
            ensure_dir(args.output_root / subset / ds_name, dry_run=args.dry_run)

    # 处理 VCTK
    vctk_counts = process_vctk(
        vctk_root=args.vctk_root,
        out_root=args.output_root,
        mode=args.mode,
        dry_run=args.dry_run,
    )

    # 处理 HiFiTTS
    hifitts_counts = process_hifitts(
        hifitts_root=args.hifitts_root,
        out_root=args.output_root,
        mode=args.mode,
        dry_run=args.dry_run,
    )

    print("\n=== 统计信息 ===")
    print("VCTK:")
    for subset in ("train", "eval", "test"):
        print(f"  {subset:5s}: {vctk_counts.get(subset, 0):6d} 音频")
    print("HiFiTTS:")
    for subset in ("train", "eval", "test"):
        print(f"  {subset:5s}: {hifitts_counts.get(subset, 0):6d} 音频")

    if args.dry_run:
        print("\n[DRY] 仅 dry-run，未实际创建任何文件。")
    else:
        print("\n混合数据集构建完成。\n"
              "你可以：\n"
              "  1) 在该输出根目录上运行 generate_captions.py + Qwen2-Audio 生成新的 caption_cache.pt；\n"
              "  2) 训练时将 --train_dir 指向 '<output_root>/train'，--val_dir 指向 '<output_root>/eval'。")


if __name__ == "__main__":
    main()
