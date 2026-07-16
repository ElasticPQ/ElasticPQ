from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and unpack MMEB media files")
    parser.add_argument("--repo-id", default="TIGER-Lab/MMEB-V2", help="Hugging Face dataset repo containing MMEB media archives")
    parser.add_argument("--output-dir", type=Path, required=True, help="Local MMEB media root")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--allow-pattern", action="append", default=[], help="Optional hf_hub_download allow pattern, repeatable")
    parser.add_argument("--ignore-pattern", action="append", default=[], help="Optional hf_hub_download ignore pattern, repeatable")
    parser.add_argument("--no-download", action="store_true", help="Only run extraction against an existing output directory")
    parser.add_argument("--no-extract", action="store_true", help="Only download archives")
    return parser.parse_args()


def download_snapshot(args: argparse.Namespace) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for MMEB media download. Install "
            "`pip install -r python/mmeb_v2_bench/requirements.txt`."
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(args.repo_id),
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.output_dir),
        allow_patterns=args.allow_pattern or None,
        ignore_patterns=args.ignore_pattern or None,
    )


def _extract_tar(path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(path), "-C", str(output_dir)], check=True)


def extract_image_tasks(root: Path) -> None:
    image_root = root / "image-tasks"
    legacy_root = root / "image_tasks"
    if not image_root.exists() and legacy_root.exists():
        image_root = legacy_root
    if not image_root.exists():
        print(f"[extract] skip image tasks; missing {root / 'image-tasks'}")
        return

    mm_root = image_root / "MMEB"
    visdoc_dir = image_root / "visdoc"
    for name in ("mmeb_v1", "visdoc"):
        archive = image_root / f"{name}.tar.gz"
        if not archive.exists():
            print(f"[extract] skip missing {archive}")
            continue
        if name == "mmeb_v1" and mm_root.exists():
            print(f"[extract] exists {mm_root}")
            continue
        target = image_root / name
        if name == "visdoc" and visdoc_dir.exists():
            print(f"[extract] exists {visdoc_dir}")
            continue
        if target.exists():
            print(f"[extract] exists {target}")
            continue
        print(f"[extract] {archive} -> {image_root}")
        _extract_tar(archive, image_root)


def extract_video_tasks(root: Path) -> None:
    frames_root = root / "video-tasks" / "frames"
    legacy_root = root / "video_tasks" / "frames"
    if not frames_root.exists() and legacy_root.exists():
        frames_root = legacy_root
    if not frames_root.exists():
        print(f"[extract] skip video tasks; missing {root / 'video-tasks' / 'frames'}")
        return

    archives = sorted(
        path
        for path in frames_root.iterdir()
        if path.is_file() and ".tar.gz" in path.name
    )
    for task in ("cls", "qa", "ret", "mret"):
        token = f"video_{task}"
        bundles = [path for path in archives if token in path.name]
        if not bundles:
            print(f"[extract] skip video_{task}; no archives")
            continue
        target = frames_root / f"video_{task}"
        if target.exists():
            print(f"[extract] exists {target}")
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"[extract] video_{task} bundles={len(bundles)} -> {target}")
        if len(bundles) == 1:
            _extract_tar(bundles[0], target)
            continue
        cat = subprocess.Popen(["cat", *map(str, bundles)], stdout=subprocess.PIPE)
        tar = subprocess.Popen(["tar", "-xzf", "-", "-C", str(target)], stdin=cat.stdout)
        if cat.stdout is not None:
            cat.stdout.close()
        cat_code = cat.wait()
        tar_code = tar.wait()
        if cat_code != 0 or tar_code != 0:
            raise RuntimeError(f"failed to extract split video_{task}: cat={cat_code} tar={tar_code}")


def extract_visdoc_tasks(root: Path) -> None:
    visdoc_root = root / "visdoc-tasks"
    legacy_root = root / "visdoc_tasks"
    if not visdoc_root.exists() and legacy_root.exists():
        visdoc_root = legacy_root
    if not visdoc_root.exists():
        print(f"[extract] skip visdoc tasks; missing {root / 'visdoc-tasks'}")
        return

    data_root = visdoc_root / "data"
    images_root = visdoc_root / "images"
    split_root = visdoc_root / "visdoc-tasks"

    for archive_name in ("visdoc-tasks.data.tar.gz", "visdoc-tasks.images.tar.gz"):
        archive = visdoc_root / archive_name
        if not archive.exists():
            print(f"[extract] skip missing {archive}")
            continue
        if archive_name == "visdoc-tasks.data.tar.gz" and data_root.exists():
            print(f"[extract] exists {data_root}")
            continue
        if archive_name == "visdoc-tasks.images.tar.gz" and images_root.exists():
            print(f"[extract] exists {images_root}")
            continue
        target = visdoc_root
        print(f"[extract] {archive} -> {target}")
        _extract_tar(archive, target)

    split_archives = sorted(path for path in visdoc_root.iterdir() if path.is_file() and "visdoc-tasks.tar.gz-" in path.name)
    if split_archives:
        if split_root.exists():
            print(f"[extract] exists {split_root}")
            return
        print(f"[extract] visdoc split bundles={len(split_archives)} -> {visdoc_root}")
        cat = subprocess.Popen(["cat", *map(str, split_archives)], stdout=subprocess.PIPE)
        tar = subprocess.Popen(["tar", "-xzf", "-", "-C", str(visdoc_root)], stdin=cat.stdout)
        if cat.stdout is not None:
            cat.stdout.close()
        cat_code = cat.wait()
        tar_code = tar.wait()
        if cat_code != 0 or tar_code != 0:
            raise RuntimeError(f"failed to extract visdoc split bundles: cat={cat_code} tar={tar_code}")


def main() -> None:
    args = _parse_args()
    if not args.no_download:
        download_snapshot(args)
    if not args.no_extract:
        extract_image_tasks(args.output_dir)
        extract_video_tasks(args.output_dir)
        extract_visdoc_tasks(args.output_dir)


if __name__ == "__main__":
    main()
