#!/usr/bin/env python3
"""
full_object_tracking_benchmark.py
---------------------------------
Benchmark SingleCameraTracker end-to-end and per-component timing vs the number
of *detected* objects on COCO-2017 val images.

It reports, per bucket (0..max_objects):
- OD time   (mean/std) in ms
- ReID time (mean/std) in ms
- Other tracker Python time (mean/std) in ms
- Total time (mean/std) in ms
- FPS mean

Notes
-----
- Requires your ODModel/ReIDModel/SingleCameraTracker wrappers.
- Works with trackers that support `timeit=True` and return (tracks, timings).
  If not, it falls back to external timing + OD/ReID estimates.

Example
-------
python full_object_tracking_benchmark.py \
  --od-model yolo --reid-model clip --tracker deep_sort \
  --weight yolov8n.pt \
  --coco-root /data/coco \
  --device cuda:0 \
  --images-per-bin 5 --repeats 50 --max-objects 20 \
  --compile-od --compile-reid
"""
from __future__ import annotations
import argparse
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CocoDetection

# --------------------------------------------------------------------------- #
#  Your wrappers                                                              #
# --------------------------------------------------------------------------- #
from branes_platform.nn.object_detection.models import ODModel
from branes_platform.nn.reid.models import ReIDModel
from branes_platform.applications.object_trackers.single_camera_tracker import SingleCameraTracker
from branes_platform.utils.timer import _Timer


# --------------------------- helpers --------------------------------------- #

def to_bgr_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """torch CHW float32 [0,1] → BGR uint8."""
    img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # RGB
    return img[:, :, ::-1]


def safe_mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), stdev(vals)

# ---------------------- bucketing by detected count ------------------------ #

def build_detect_buckets(
    dataset: CocoDetection,
    od_model: ODModel,
    max_objects: int,
    images_per_bin: int,
    conf_thres: float,
) -> Dict[int, List[int]]:
    """One OD forward per image to group indices by DETECTED object count."""
    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx in tqdm(range(len(dataset)), desc="Scanning images"):
        if all(len(buckets[k]) >= images_per_bin for k in range(max_objects + 1)):
            break
        img_tensor, _ = dataset[idx]
        dets = od_model.predict(to_bgr_uint8(img_tensor), conf_thres=conf_thres)
        n = int(dets.shape[0])
        if n > max_objects:
            continue
        if len(buckets[n]) < images_per_bin:
            buckets[n].append(idx)
    return buckets

# ---------------------- per-image component timers ------------------------- #

def measure_od_time_ms(sct: SingleCameraTracker, img_bgr: np.ndarray, conf_thres: float) -> float:
    """Time the OD forward only, using the model/device in `sct`."""
    t = _Timer(sct.od.device)
    t.start()
    _ = sct.od.predict(img_bgr, conf_thres=conf_thres)
    return t.stop_ms()

def measure_reid_time_ms(sct: SingleCameraTracker, img_bgr: np.ndarray, conf_thres: float) -> float:
    """Time the ReID forward only, batching all det crops detected by sct.od."""
    dets = sct.od.predict(img_bgr, conf_thres=conf_thres)
    if dets.numel() == 0:
        return 0.0
    boxes = dets[:, :4]
    t = _Timer(sct.reid.device)
    t.start()
    _ = sct.reid.predict(img_bgr, boxes)
    return t.stop_ms()

def time_tracker_on_image(
    img_bgr: np.ndarray,
    tracker_maker,
    repeats: int,
    warmup: int,
    conf_thres: float,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Returns lists (all length == repeats):
      od_ms_list, reid_ms_list, other_ms_list, total_ms_list
    FPS can be derived as 1000/total_ms.
    """
    sct: SingleCameraTracker = tracker_maker(timeit=True)

    # Warmup
    for _ in range(warmup):
        out = sct.update(img_bgr, conf_thres=conf_thres) if "conf_thres" in sct.update.__code__.co_varnames else sct.update(img_bgr)
        # ensure tuple shape not required during warmup

    od_ms_list: List[float] = []
    reid_ms_list: List[float] = []
    other_ms_list: List[float] = []
    total_ms_list: List[float] = []

    # Try to use model-provided timing if available
    returns_timings = False

    for _ in range(repeats):
        # If SingleCameraTracker.update returns timings when timeit=True, use them
        t_total = _Timer("cpu"); t_total.start()
        result = sct.update(img_bgr, conf_thres=conf_thres) if "conf_thres" in sct.update.__code__.co_varnames else sct.update(img_bgr)
        total_ms_measured = t_total.stop_ms()

        od_ms = None
        reid_ms = None
        other_ms = None
        total_ms = total_ms_measured

        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            returns_timings = True
            timings = result[1]
            # prefer model-provided
            total_ms = float(timings.get("total_ms", total_ms_measured))
            od_ms = float(timings.get("od_ms", np.nan))
            reid_ms = float(timings.get("reid_ms", np.nan))
            # compute other if both known
            if np.isfinite(od_ms) and np.isfinite(reid_ms):
                other_ms = max(0.0, total_ms - od_ms - reid_ms)
        # Fallback: estimate components externally if not provided
        if not returns_timings or od_ms is None or not np.isfinite(od_ms):
            od_ms = measure_od_time_ms(sct, img_bgr, conf_thres)
        if not returns_timings or reid_ms is None or not np.isfinite(reid_ms):
            reid_ms = measure_reid_time_ms(sct, img_bgr, conf_thres)
        if other_ms is None:
            other_ms = max(0.0, total_ms - od_ms - reid_ms)

        od_ms_list.append(float(od_ms))
        reid_ms_list.append(float(reid_ms))
        other_ms_list.append(float(other_ms))
        total_ms_list.append(float(total_ms))

    return od_ms_list, reid_ms_list, other_ms_list, total_ms_list

# --------------------------- main routine ---------------------------------- #

def run_benchmark(
    od_model_name: str,
    reid_model_name: str,
    tracker_name: str,
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
    tracker_kwargs: dict[str, Any],
):
    # Dataset
    tfm = transforms.Compose([transforms.ToTensor()])
    root = Path(coco_root)
    ds = CocoDetection(
        root=root / "val2017",
        annFile=root / "annotations" / "instances_val2017.json",
        transform=tfm,
    )

    # OD for bucketing
    bucket_od = ODModel(
        model_name=od_model_name,
        device=device_str,
        compile_model=compile_od,
        **({"weight": weight} if weight else {}),
    )
    bucket_od.model.eval()

    print("[1/3] Scanning & building buckets ...")
    buckets = build_detect_buckets(
        dataset=ds,
        od_model=bucket_od,
        max_objects=max_objects,
        images_per_bin=images_per_bin,
        conf_thres=conf_thres,
    )

    # Factory for fresh trackers (for timing stage)
    def make_tracker(timeit: bool = False) -> SingleCameraTracker:
        return SingleCameraTracker(
            od_name=od_model_name,
            reid_name=reid_model_name,
            sort_algorithm=tracker_name,
            compile_od=compile_od,
            compile_reid=compile_reid,
            od_kwargs=({"weight": weight} if weight else {}),
            tracker_kwargs={"timeit": timeit, **(tracker_kwargs or {})},
            device=device_str,
            # and pass timeit down to SCT so it can propagate to OD/ReID and tracker
            timeit=timeit,   # assuming you added this flag to SCT __init__
        )

    per_bucket_stats: Dict[int, Dict[str, List[float]]] = {}

    print("[2/3] Timing per bucket ...")
    for obj_cnt, idx_list in buckets.items():
        od_all: List[float] = []
        reid_all: List[float] = []
        other_all: List[float] = []
        total_all: List[float] = []

        for idx in idx_list:
            img_tensor, _ = ds[idx]
            img_bgr = to_bgr_uint8(img_tensor)

            od_ms, reid_ms, other_ms, total_ms = time_tracker_on_image(
                img_bgr,
                tracker_maker=make_tracker,
                repeats=repeats,
                warmup=warmup_iters,
                conf_thres=conf_thres,
            )
            od_all.extend(od_ms)
            reid_all.extend(reid_ms)
            other_all.extend(other_ms)
            total_all.extend(total_ms)

        per_bucket_stats[obj_cnt] = {
            "od_ms": od_all,
            "reid_ms": reid_all,
            "other_ms": other_all,
            "total_ms": total_all,
        }

    # ---------------- report ------------------------------------------------ #
    print("\n[3/3] Report (per detected-object bucket)")
    hdr = (
        f"{'objects':>7} | {'samples':>7} | "
        f"{'OD ms (μ/σ)':>14} | {'ReID ms (μ/σ)':>16} | "
        f"{'Other ms (μ/σ)':>16} | {'Total ms (μ/σ)':>16} | {'FPS μ':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    for obj_cnt in sorted(per_bucket_stats):
        stats = per_bucket_stats[obj_cnt]
        n = len(stats["total_ms"])

        od_mu, od_sd = safe_mean_std(stats["od_ms"])
        reid_mu, reid_sd = safe_mean_std(stats["reid_ms"])
        oth_mu, oth_sd = safe_mean_std(stats["other_ms"])
        tot_mu, tot_sd = safe_mean_std(stats["total_ms"])
        fps_mu = (1000.0 / tot_mu) if tot_mu > 0 else 0.0

        print(
            f"{obj_cnt:>7} | {n:>7} | "
            f"{od_mu:>7.2f}/{od_sd:>5.2f} | "
            f"{reid_mu:>7.2f}/{reid_sd:>5.2f} | "
            f"{oth_mu:>7.2f}/{oth_sd:>5.2f} | "
            f"{tot_mu:>7.2f}/{tot_sd:>5.2f} | "
            f"{fps_mu:>8.2f}"
        )

# --------------------------- CLI ------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Full object tracking benchmark")
    p.add_argument("--od-model", default="yolo",
                   help="Your ODModel name (e.g., yolo, detr, rtdetr, yolos, fasterrcnn, ssd300)")
    p.add_argument("--reid-model", default="clip",
                   help="Your ReIDModel name (e.g., clip, clip_vit_b32, dinov2_vits14, osnet, mobilenetv2, resnet18)")
    p.add_argument("--tracker", default="deep_sort",
                   choices=["deep_sort", "oc_sort", "bot_sort", "bytetrack", "sort"],
                   help="Tracker algorithm")
    p.add_argument("--weight", default=None, help="OD checkpoint (optional)")
    p.add_argument("--coco-root", required=True, help="Path to COCO 2017 root")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--conf-thres", type=float, default=0.25)
    p.add_argument("--images-per-bin", type=int, default=5,
                   help="Max images sampled per object-count bin")
    p.add_argument("--repeats", type=int, default=50,
                   help="update() calls per image (timed)")
    p.add_argument("--max-objects", type=int, default=20)
    p.add_argument("--warmup-iters", type=int, default=3)
    p.add_argument("--compile-od", action="store_true")
    p.add_argument("--compile-reid", action="store_true")
    # tracker-specific overrides if you want to experiment without code changes
    p.add_argument("--tracker-kwargs", default="{}", help='JSON dict, e.g. \'{"match_iou":0.3}\'')
    return p.parse_args()

def _parse_json_dict(s: str) -> dict:
    import json
    try:
        d = json.loads(s)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

if __name__ == "__main__":
    a = parse_args()
    run_benchmark(
        od_model_name=a.od_model,
        reid_model_name=a.reid_model,
        tracker_name=a.tracker,
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
        tracker_kwargs=_parse_json_dict(a.tracker_kwargs),
    )