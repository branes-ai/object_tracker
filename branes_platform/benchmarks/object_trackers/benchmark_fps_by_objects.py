#!/usr/bin/env python3
"""
benchmark_fps_by_objects.py
---------------------------------
Compute inference FPS vs. number-of-objects-in-image for ODModel
(YOLO-v8/9 or DETR) on the COCO-2017 validation split.

Example
-------
python benchmark_fps_by_objects.py \
       --model yolo --weight yolov8n.pt \
       --coco-root /data/coco \
       --device cuda:0 \
       --samples-per-bin 100 \
       --max-objects 30
"""

from __future__ import annotations
import argparse
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CocoDetection

# --------------------------------------------------------------------------- #
#  Import your wrapper                                                        #
# --------------------------------------------------------------------------- #
from branes_platform.nn.object_detection.deep_learning_models import ODModel     # EDIT as needed
from branes_platform.applications.object_trackers.single_camera_tracker import SingleCameraTracker


def to_bgr_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """torch CHW float32 [0,1] → BGR uint8."""
    img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)   # RGB
    return img[:, :, ::-1]


def build_detect_buckets(
    dataset: CocoDetection,
    od_model: ODModel,
    max_objects: int,
    images_per_bin: int,
    conf_thres: float,
) -> Dict[int, List[int]]:
    """
    One forward pass through ODModel to group image indices by *detected*
    object count.  Stops early when every bucket 0…max_objects is full.
    """
    buckets: Dict[int, List[int]] = defaultdict(list)

    for idx in tqdm(range(len(dataset)), desc="Scanning images"):
        if all(len(buckets[k]) >= images_per_bin for k in range(max_objects + 1)):
            break  # already filled all buckets
        img_tensor, _ = dataset[idx]
        dets = od_model.predict(to_bgr_uint8(img_tensor), conf_thres=conf_thres)
        n = dets.shape[0]
        if n > max_objects:
            continue
        if len(buckets[n]) < images_per_bin:
            buckets[n].append(idx)
    return buckets


def time_tracker_on_image(
    img_bgr: np.ndarray,
    tracker_maker,
    repeats: int,
    warmup: int,
) -> List[float]:
    """Return list of FPS values (len == repeats) for one image."""
    sct = tracker_maker()
    for _ in range(warmup):
        _ = sct.update(img_bgr)

    fps_vals: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = sct.update(img_bgr)
        dt = time.perf_counter() - t0
        fps_vals.append(1.0 / dt if dt > 0 else 0.0)
    return fps_vals


# --------------------- master routine -------------------------------------- #
def run_benchmark(
    od_model_name: str,
    reid_model_name: str,
    weight: str | None,
    coco_root: str,
    device_str: str,
    conf_thres: float,
    images_per_bin: int,
    repeats: int,
    max_objects: int,
    warmup_iters: int,
    compile_od: bool,
    compile_reid: bool,
):
    # Dataset
    tfm = transforms.Compose([transforms.ToTensor()])
    root = Path(coco_root)
    ds = CocoDetection(
        root=root / "val2017",
        annFile=root / "annotations" / "instances_val2017.json",
        transform=tfm,
    )

    # Model for *bucketing* (just OD, single forward each image)
    bucket_od = ODModel(
        model_name=od_model_name,
        device=device_str,
        compile_model=compile_od,
        **({"weight": weight} if weight else {}),
    )
    bucket_od.model.eval()

    buckets = build_detect_buckets(
        dataset=ds,
        od_model=bucket_od,
        max_objects=max_objects,
        images_per_bin=images_per_bin,
        conf_thres=conf_thres,
    )

    # Factory for fresh trackers (used in timing stage)
    def make_tracker():
        return SingleCameraTracker(
            od_name=od_model_name,
            reid_name=reid_model_name,
            compile_od=compile_od,
            compile_reid=compile_reid,
            od_kwargs=({"weight": weight} if weight else {}),
            device=device_str,
        )

    results: Dict[int, List[float]] = {}

    for obj_cnt, idx_list in buckets.items():
        all_fps: List[float] = []
        for idx in idx_list:
            img_tensor, _ = ds[idx]
            img_bgr = to_bgr_uint8(img_tensor)

            fps_vals = time_tracker_on_image(
                img_bgr,
                tracker_maker=make_tracker,
                repeats=repeats,
                warmup=warmup_iters,
            )
            all_fps.extend(fps_vals)

        results[obj_cnt] = all_fps

    # ------------ report ---------------------------------------------------- #
    print(f"\n--- SCT FPS vs. *detected* object count "
          f"({images_per_bin} imgs × {repeats} reps) ---")
    print(f"{'objects':>7} | {'samples':>7} | {'FPS mean':>8} | {'FPS std':>8}")
    print("-" * 45)
    for obj_cnt in sorted(results):
        fps = results[obj_cnt]
        print(f"{obj_cnt:>7} | {len(fps):>7} | {mean(fps):>8.2f} | "
              f"{(stdev(fps) if len(fps) > 1 else 0):>8.2f}")


# --------------------- CLI -------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Benchmark SCT FPS vs detected objects")
    p.add_argument("--od-model", default="yolo", choices=["yolo", "detr"])
    p.add_argument("--reid-model", default="clip")
    p.add_argument("--weight", default=None, help="OD checkpoint (optional)")
    p.add_argument("--coco-root", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--conf-thres", type=float, default=0.25)
    p.add_argument("--images-per-bin", type=int, default=5,
                   help="Max images sampled per object-count bin")
    p.add_argument("--repeats", type=int, default=100,
                   help="update() calls per image (timed)")
    p.add_argument("--max-objects", type=int, default=30)
    p.add_argument("--warmup-iters", type=int, default=3)
    p.add_argument("--compile-od", action="store_true")
    p.add_argument("--compile-reid", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_benchmark(
        od_model_name=a.od_model,
        reid_model_name=a.reid_model,
        weight=a.weight,
        coco_root=a.coco_root,
        device_str=a.device,
        conf_thres=a.conf_thres,
        images_per_bin=a.images_per_bin,
        repeats=a.repeats,
        max_objects=a.max_objects,
        warmup_iters=a.warmup_iters,
        compile_od=a.compile_od,
        compile_reid=a.compile_reid,
    )