#!/usr/bin/env python3
"""
benchmark_sct_fps.py
--------------------
Measure *tracking* FPS (SingleCameraTracker) versus the number of
ground-truth objects in a COCO-2017-val image.

For each object-count bucket (0, 1, 2 … `--max-objects`) the script:

1. Randomly picks **one** image from that bucket.
2. Runs `SingleCameraTracker.update()` on *the same image*
   `--repeats` times (stateful tracker, so tracks accumulate).
3. Reports the mean / std FPS for that bucket.

Example
-------
python benchmark_sct_fps.py \
       --od-model yolo --reid-model clip \
       --weight yolov8n.pt \
       --coco-root /data/coco \
       --device cuda:0 \
       --repeats 100 \
       --compile-od --compile-reid
"""

from __future__ import annotations
import argparse
import random
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CocoDetection

# --------------------------------------------------------------------------- #
#  your tracker wrapper                                                       #
# --------------------------------------------------------------------------- #
from branes_platform.applications.object_trackers.single_camera_tracker import SingleCameraTracker  # adjust if needed


# --------------------- helpers --------------------------------------------- #
def to_bgr_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """torch.Tensor CHW float32 [0,1] → BGR uint8"""
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # RGB
    return img_np[:, :, ::-1]


def bucket_indices_by_objects(ds: CocoDetection) -> Dict[int, List[int]]:
    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(ds)):
        _, target = ds[idx]
        buckets[len(target)].append(idx)
    return buckets


# --------------------- main routine ---------------------------------------- #
def benchmark_bucket(
    img_bgr: np.ndarray,
    tracker_ctor,
    repeats: int,
    warmup: int,
) -> List[float]:
    """Run one bucket; return list of FPS values (len == repeats)."""
    sct = tracker_ctor()  # fresh tracker for this bucket
    # warm-up
    for _ in range(warmup):
        _ = sct.update(img_bgr)

    fps_list: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = sct.update(img_bgr)
        dt = time.perf_counter() - t0
        fps_list.append(1.0 / dt if dt > 0 else 0.0)
    return fps_list


def run_benchmark(
    od_model: str,
    reid_model: str,
    weight: str | None,
    coco_root: str,
    device_str: str,
    repeats: int,
    max_objects: int,
    warmup_iters: int,
    compile_od: bool,
    compile_reid: bool,
):
    # dataset
    trans = transforms.Compose([transforms.ToTensor()])
    root = Path(coco_root)
    ds = CocoDetection(
        root=root / "val2017",
        annFile=root / "annotations" / "instances_val2017.json",
        transform=trans,
    )
    buckets = bucket_indices_by_objects(ds)

    # factory to create a fresh tracker each bucket
    def make_tracker():
        return SingleCameraTracker(
            od_name=od_model,
            reid_name=reid_model,
            compile_od=compile_od,
            compile_reid=compile_reid,
            od_kwargs=({"weight": weight} if weight else {}),
            device=device_str,
        )

    results: Dict[int, List[float]] = {}

    for obj_cnt in range(0, max_objects + 1):
        if obj_cnt not in buckets:
            continue
        idx = random.choice(buckets[obj_cnt])
        img_tensor, _ = ds[idx]
        img_bgr = to_bgr_uint8(img_tensor)

        fps_vals = benchmark_bucket(
            img_bgr,
            tracker_ctor=make_tracker,
            repeats=repeats,
            warmup=warmup_iters,
        )
        results[obj_cnt] = fps_vals

    # report
    print(f"\n--- SingleCameraTracker FPS ({repeats} repeats) ---")
    print(f"{'objects':>7} | {'FPS mean':>8} | {'FPS std':>8}")
    print("-" * 30)
    for obj_cnt in sorted(results):
        fps = results[obj_cnt]
        print(f"{obj_cnt:>7} | {mean(fps):>8.2f} | "
              f"{(stdev(fps) if len(fps) > 1 else 0):>8.2f}")


# --------------------- CLI -------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Benchmark SCT FPS vs. object count")
    p.add_argument("--od-model", default="yolo", choices=["yolo", "detr"])
    p.add_argument("--reid-model", default="clip")
    p.add_argument("--weight", default=None, help="OD checkpoint (optional)")
    p.add_argument("--coco-root", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--repeats", type=int, default=100,
                   help="How many consecutive update() calls per bucket")
    p.add_argument("--max-objects", type=int, default=30)
    p.add_argument("--warmup-iters", type=int, default=3)
    p.add_argument("--compile-od", action="store_true")
    p.add_argument("--compile-reid", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_benchmark(
        od_model=a.od_model,
        reid_model=a.reid_model,
        weight=a.weight,
        coco_root=a.coco_root,
        device_str=a.device,
        repeats=a.repeats,
        max_objects=a.max_objects,
        warmup_iters=a.warmup_iters,
        compile_od=a.compile_od,
        compile_reid=a.compile_reid,
    )