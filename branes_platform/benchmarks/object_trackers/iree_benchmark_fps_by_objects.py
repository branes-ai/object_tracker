#!/usr/bin/env python3
"""
benchmark_fps_by_objects_iree.py
--------------------------------
Compute SCT (YOLO/CLIP over IREE) FPS vs. number-of-objects-in-image
on COCO-2017 val. Uses ODModelIREE for bucketing and SingleCameraTrackerIREE
for timing.

Example
-------
python benchmark_fps_by_objects_iree.py \
  --od-vmfb yolov8n.vmfb \
  --reid-vmfb clip_vitb32_visual_cpu.vmfb \
  --coco-root /data/coco \
  --iree-driver local-task \
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
#  IREE wrappers                                                              #
# --------------------------------------------------------------------------- #
from branes_platform.nn.object_detection.iree_model import ODModelIREE
from branes_platform.applications.object_trackers.single_camera_tracker import SingleCameraTrackerIREE

# --------------------------------------------------------------------------- #

def to_bgr_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """torch CHW float32 [0,1] → BGR uint8."""
    img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)   # RGB
    return img[:, :, ::-1]  # BGR


def build_detect_buckets(
    dataset: CocoDetection,
    od_model: ODModelIREE,
    max_objects: int,
    images_per_bin: int,
    conf_thres: float,
) -> Dict[int, List[int]]:
    """
    One forward pass through ODModelIREE to group image indices by *detected*
    object count. Stops early when every bucket 0…max_objects is full.
    """
    buckets: Dict[int, List[int]] = defaultdict(list)

    for idx in tqdm(range(len(dataset)), desc="Scanning images"):
        if all(len(buckets[k]) >= images_per_bin for k in range(max_objects + 1)):
            break  # already filled all buckets

        img_tensor, _ = dataset[idx]
        dets = od_model.predict(to_bgr_uint8(img_tensor))
        # Robust: filter by conf here in case ODModelIREE doesn't filter internally
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy()
        if dets.size:
            dets = dets[dets[:, 4] >= conf_thres]
        n = dets.shape[0] if dets.size else 0

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
    od_vmfb: str,
    reid_vmfb: str,
    coco_root: str,
    iree_driver: str,
    conf_thres: float,
    images_per_bin: int,
    repeats: int,
    max_objects: int,
    warmup_iters: int,
):
    # Dataset
    tfm = transforms.Compose([transforms.ToTensor()])
    root = Path(coco_root)
    ds = CocoDetection(
        root=root / "val2017",
        annFile=root / "annotations" / "instances_val2017.json",
        transform=tfm,
    )

    # Model for *bucketing* (OD only, single forward per image)
    bucket_od = ODModelIREE(od_vmfb, device=iree_driver)

    buckets = build_detect_buckets(
        dataset=ds,
        od_model=bucket_od,
        max_objects=max_objects,
        images_per_bin=images_per_bin,
        conf_thres=conf_thres,
    )

    # Factory for fresh IREE trackers (used in timing stage)
    def make_tracker():
        return SingleCameraTrackerIREE(
            od_vmfb=od_vmfb,
            reid_vmfb=reid_vmfb,
            tracker_kwargs=None,
            device=iree_driver,
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
    print(f"\n--- SCT (IREE) FPS vs. *detected* object count "
          f"({images_per_bin} imgs × {repeats} reps) ---")
    print(f"{'objects':>7} | {'samples':>7} | {'FPS mean':>8} | {'FPS std':>8}")
    print("-" * 45)
    for obj_cnt in sorted(results):
        fps = results[obj_cnt]
        print(f"{obj_cnt:>7} | {len(fps):>7} | {mean(fps):>8.2f} | "
              f"{(stdev(fps) if len(fps) > 1 else 0):>8.2f}")


# --------------------- CLI -------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Benchmark SCT (IREE) FPS vs detected objects")
    p.add_argument("--od-vmfb", required=True, help="Path to YOLO/DETR IREE vmfb")
    p.add_argument("--reid-vmfb", required=True, help="Path to CLIP visual IREE vmfb")
    p.add_argument("--coco-root", required=True)
    p.add_argument("--iree-driver", default="cpu",
                   help="IREE driver: local-task | cuda | vulkan | metal")
    p.add_argument("--conf-thres", type=float, default=0.25)
    p.add_argument("--images-per-bin", type=int, default=5,
                   help="Max images sampled per object-count bin")
    p.add_argument("--repeats", type=int, default=100,
                   help="update() calls per image (timed)")
    p.add_argument("--max-objects", type=int, default=30)
    p.add_argument("--warmup-iters", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_benchmark(
        od_vmfb=a.od_vmfb,
        reid_vmfb=a.reid_vmfb,
        coco_root=a.coco_root,
        iree_driver=a.iree_driver,
        conf_thres=a.conf_thres,
        images_per_bin=a.images_per_bin,
        repeats=a.repeats,
        max_objects=a.max_objects,
        warmup_iters=a.warmup_iters,
    )