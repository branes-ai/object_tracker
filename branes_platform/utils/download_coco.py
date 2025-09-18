#!/usr/bin/env python3
"""
download_coco.py
----------------
Utility to auto-download and extract the COCO 2017 validation dataset
(images + annotations).

Images (~1 GB):   http://images.cocodataset.org/zips/val2017.zip
Annotations (~240 MB): http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Output structure (under --root):
    root/
      ├── val2017/
      └── annotations/
            └── instances_val2017.json

Example
-------
python download_coco.py --root /data/coco
"""

from __future__ import annotations
import argparse
from pathlib import Path
from torchvision.datasets.utils import download_and_extract_archive

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_coco(root: Path):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if not (root / "val2017").exists():
        print("Downloading COCO val2017 images (~1 GB)…")
        download_and_extract_archive(COCO_VAL_URL, download_root=root, extract_root=root)

    anns = root / "annotations" / "instances_val2017.json"
    if not anns.exists():
        print("Downloading COCO annotations (~240 MB)…")
        download_and_extract_archive(COCO_ANNS_URL, download_root=root, extract_root=root)

    print(f"✅ COCO 2017 val dataset ready under: {root}")


def parse_args():
    p = argparse.ArgumentParser(description="Download COCO 2017 validation dataset")
    p.add_argument("--root", type=str, required=True, help="Target directory for dataset")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_coco(Path(args.root))