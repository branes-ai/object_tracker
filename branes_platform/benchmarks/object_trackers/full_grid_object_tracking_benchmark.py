#!/usr/bin/env python3
"""
full_grid_object_tracking_benchmark.py
--------------------------------------
Cycle through trackers, OD models, and (if applicable) ReID models; benchmark
per object-count bin on COCO-2017 val split.

It reports mean/std for:
- OD time (ms)
- ReID time (ms) [0 if tracker doesn't use ReID]
- Other time (ms) = Total - OD - ReID
- Total time (ms)
- FPS (1 / Total)

Also prints a table and saves a CSV.

Example
-------
python full_grid_object_tracking_benchmark.py \
  --coco-root /data/coco \
  --device cuda:0 \
  --conf-thres 0.25 \
  --images-per-bin 5 \
  --repeats 30 \
  --start 0 --end 20 --step 2 \
  --csv-out benchmark_output.csv

# Run a single configuration only:
python full_grid_object_tracking_benchmark.py \
  --tracker deep_sort --od-model yolo --reid-model clip \
  --coco-root /data/coco --device cuda:0
"""

from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Iterable, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CocoDetection

# Your wrappers
from branes_platform.nn.object_detection.models import ODModel
from branes_platform.nn.reid.models import ReIDModel
from branes_platform.applications.object_trackers.single_camera_tracker import SingleCameraTracker


# -------------------------- supported registries ---------------------------- #

SUPPORTED_OD = ["yolo", "detr", "rt-detr", "yolos", "fasterrcnn", "ssd300"]

# trackers that DO / DO NOT use ReID
TRACKERS_WITH_REID = {"deep_sort", "bot_sort"}
TRACKERS_NO_REID   = {"oc_sort", "bytetrack", "sort"}
SUPPORTED_TRACKERS = sorted(list(TRACKERS_WITH_REID | TRACKERS_NO_REID))

SUPPORTED_REID = ["clip", "clip_vit_b32", "dinov2_vits14", "osnet_x0_25", "mobilenetv2", "resnet18"]


# ------------------------------ utils -------------------------------------- #

def to_bgr_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """torch CHW float32 [0,1] → BGR uint8."""
    img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)   # RGB
    return img[:, :, ::-1]


def download_coco_if_missing(root: Path) -> None:
    """Optional auto-download if missing (val2017 + annotations)."""
    COCO_VAL_URL   = "http://images.cocodataset.org/zips/val2017.zip"
    COCO_ANNS_URL  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "val2017").exists():
        print("Downloading COCO val2017 images (~1 GB)…")
        from torchvision.datasets.utils import download_and_extract_archive
        download_and_extract_archive(COCO_VAL_URL, download_root=root, extract_root=root)
    anns = root / "annotations" / "instances_val2017.json"
    if not anns.exists():
        print("Downloading COCO annotations (~240 MB)…")
        from torchvision.datasets.utils import download_and_extract_archive
        download_and_extract_archive(COCO_ANNS_URL, download_root=root, extract_root=root)


def build_detect_buckets_for_od(
    dataset: CocoDetection,
    od_model: ODModel,
    needed_bins: Iterable[int],
    images_per_bin: int,
    conf_thres: float,
) -> Dict[int, List[int]]:
    """
    One pass over the dataset *with the given OD model* to group image indices
    by detected object count. Only fills requested bins in `needed_bins`.
    """
    needed = set(needed_bins)
    buckets: Dict[int, List[int]] = {k: [] for k in needed}

    for idx in tqdm(range(len(dataset)), desc=f"Scanning images for OD={od_model.model_name}"):
        if all(len(buckets[k]) >= images_per_bin for k in needed):
            break
        img_tensor, _ = dataset[idx]
        dets = od_model.predict(to_bgr_uint8(img_tensor), conf_thres=conf_thres)
        n = dets.shape[0]
        if n in needed and len(buckets[n]) < images_per_bin:
            buckets[n].append(idx)

    # warn if any bins are empty
    for k in sorted(needed):
        if len(buckets[k]) == 0:
            print(f"[warn] No images found for bin objects={k} with OD='{od_model.model_name}'.")
    return buckets


@dataclass
class Timings:
    od_ms:   List[float]
    reid_ms: List[float]
    tot_ms:  List[float]

    @property
    def other_ms(self) -> List[float]:
        # clamp at zero to avoid small negative due to timer jitter
        return [max(0.0, t - o - r) for t, o, r in zip(self.tot_ms, self.od_ms, self.reid_ms)]

    @staticmethod
    def empty() -> "Timings":
        return Timings([], [], [])


def time_one_iteration(
    img_bgr: np.ndarray,
    sct: SingleCameraTracker,
    conf_thres: float,
    tracker_uses_reid: bool,
) -> Tuple[float, float, float]:
    """
    Time OD, ReID (optional), and Total (end-to-end update) on the same image.
    Returns (od_ms, reid_ms, total_ms).
    """
    # OD timing
    t0 = time.perf_counter()
    dets = sct.od.predict(img_bgr, conf_thres=conf_thres)
    od_ms = (time.perf_counter() - t0) * 1000.0

    # ReID timing (if tracker uses it)
    if tracker_uses_reid and dets.numel():
        boxes_xyxy = dets[:, :4]
        t1 = time.perf_counter()
        _ = sct.reid.predict(img_bgr, boxes_xyxy)
        reid_ms = (time.perf_counter() - t1) * 1000.0
    else:
        reid_ms = 0.0

    # Total timing (tracker.update internally re-runs OD/ReID)
    t2 = time.perf_counter()
    _ = sct.update(img_bgr)  # end-to-end
    tot_ms = (time.perf_counter() - t2) * 1000.0

    return od_ms, reid_ms, tot_ms


def summarize_timings(ts: Timings) -> Dict[str, float]:
    def m(v): return mean(v) if v else 0.0
    def s(v): return stdev(v) if len(v) > 1 else 0.0
    other = ts.other_ms
    fps  = [1000.0 / t for t in ts.tot_ms if t > 0]
    return {
        "od_mean_ms":   m(ts.od_ms),   "od_std_ms":   s(ts.od_ms),
        "reid_mean_ms": m(ts.reid_ms), "reid_std_ms": s(ts.reid_ms),
        "other_mean_ms": m(other),     "other_std_ms": s(other),
        "total_mean_ms": m(ts.tot_ms), "total_std_ms": s(ts.tot_ms),
        "fps_mean":     m(fps),
    }


# ------------------------------- main run ----------------------------------- #

def run_benchmark_grid(
    *,
    trackers: List[str],
    od_models: List[str],
    reid_models: List[str],
    weight: Optional[str],
    coco_root: str,
    device_str: str,
    conf_thres: float,
    images_per_bin: int,
    repeats: int,
    warmup_iters: int,
    compile_od: bool,
    compile_reid: bool,
    start_bin: int,
    end_bin: int,
    step_bin: int,
    csv_out: Optional[str],
    tracker_kwargs_json: str,
):
    start_time = dt.datetime.now()

    root = Path(coco_root)
    if not (root / "val2017").exists() or not (root / "annotations" / "instances_val2017.json").exists():
        print("[info] COCO not found; attempting auto-download…")
        download_coco_if_missing(root)

    ds = CocoDetection(
        root=root / "val2017",
        annFile=root / "annotations" / "instances_val2017.json",
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    object_bins = list(range(start_bin, end_bin + 1, step_bin))
    tracker_overrides = {}
    try:
        tracker_overrides = json.loads(tracker_kwargs_json) if tracker_kwargs_json else {}
    except Exception as e:
        print(f"[warn] Failed to parse --tracker-kwargs; using defaults. Error: {e}")

    rows = []  # accumulate CSV rows

    # Precompute combo list for a clean tqdm
    combos = []
    for od_name in od_models:
        for tracker_name in trackers:
            uses_reid = tracker_name in TRACKERS_WITH_REID
            reid_list = (reid_models if uses_reid else [None])
            for reid_name in reid_list:
                combos.append((od_name, tracker_name, reid_name))

    combo_bar = tqdm(combos, desc="Config combos", position=0)

    for od_name, tracker_name, reid_name in combo_bar:
        combo_bar.set_postfix_str(f"OD={od_name} | TR={tracker_name} | REID={(reid_name or '-')}")
        # Prepare a lightweight OD model for bucketing
        bucket_od = ODModel(
            model_name=od_name,
            device=device_str,
            compile_model=compile_od,
            **({"weight": weight} if weight else {}),
        )
        bucket_od.model.eval()

        buckets = build_detect_buckets_for_od(
            dataset=ds,
            od_model=bucket_od,
            needed_bins=object_bins,
            images_per_bin=images_per_bin,
            conf_thres=conf_thres,
        )

        uses_reid = tracker_name in TRACKERS_WITH_REID

        # estimate total steps for progress within this combo
        total_images = sum(len(buckets.get(k, [])) for k in object_bins)
        total_steps = total_images * (warmup_iters + repeats)
        inner_bar = tqdm(total=total_steps, desc=f"Run {od_name}/{tracker_name}/{reid_name or '-'}", position=1, leave=False)

        # factory for SCT
        def make_sct() -> SingleCameraTracker:
            return SingleCameraTracker(
                od_name=od_name,
                reid_name=(reid_name or "clip"),
                sort_algorithm=tracker_name,
                compile_od=compile_od,
                compile_reid=compile_reid,
                od_kwargs=({"weight": weight} if weight else {}),
                tracker_kwargs=tracker_overrides,
                device=device_str,
                timeit=True
            )

        for obj_cnt in object_bins:
            idx_list = buckets.get(obj_cnt, [])
            if len(idx_list) == 0:
                continue

            all_ts = Timings.empty()

            for idx in idx_list:
                img_tensor, _ = ds[idx]
                img_bgr = to_bgr_uint8(img_tensor)

                sct = make_sct()

                # warmup (not recorded)
                for _ in range(warmup_iters):
                    _ = sct.update(img_bgr)
                    inner_bar.update(1)

                # timed repeats
                for _ in range(repeats):
                    od_ms, reid_ms, tot_ms = time_one_iteration(
                        img_bgr, sct, conf_thres, uses_reid
                    )
                    all_ts.od_ms.append(od_ms)
                    all_ts.reid_ms.append(reid_ms)
                    all_ts.tot_ms.append(tot_ms)
                    inner_bar.update(1)

            stats = summarize_timings(all_ts)
            row = {
                "tracker": tracker_name,
                "od": od_name,
                "reid": (reid_name or ""),
                "objects": obj_cnt,
                "samples": len(idx_list) * repeats,
                **stats,
            }
            rows.append(row)

            # print one line per bin
            print(
                f"[{tracker_name:9s}] od={od_name:10s} "
                f"reid={(reid_name or '-'):12s} "
                f"objs={obj_cnt:2d}  samples={row['samples']:4d} | "
                f"OD {row['od_mean_ms']:.1f}±{row['od_std_ms']:.1f} ms  "
                f"ReID {row['reid_mean_ms']:.1f}±{row['reid_std_ms']:.1f} ms  "
                f"Other {row['other_mean_ms']:.1f}±{row['other_std_ms']:.1f} ms  "
                f"Total {row['total_mean_ms']:.1f}±{row['total_std_ms']:.1f} ms  "
                f"FPS {row['fps_mean']:.1f}"
            )

        inner_bar.close()

    # save CSV
    if not rows:
        print("\n[warn] No results collected. Check your bins or dataset availability.")
    end_time = dt.datetime.now()

    if not csv_out:
        ts = end_time.strftime("%Y%m%d_%H%M%S")
        csv_out = f"benchmark_output_{ts}.csv"
    csv_path = Path(csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "tracker", "od", "reid", "objects", "samples",
        "od_mean_ms", "od_std_ms",
        "reid_mean_ms", "reid_std_ms",
        "other_mean_ms", "other_std_ms",
        "total_mean_ms", "total_std_ms",
        "fps_mean",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nSaved CSV to: {csv_path.resolve()}")

    # Final timing summary
    elapsed = end_time - start_time
    def fmt(t: dt.datetime) -> str:
        return t.strftime("%Y-%m-%d %H:%M:%S")
    print("\n=== Benchmark timing summary ===")
    print(f"Started: {fmt(start_time)}")
    print(f"Ended  : {fmt(end_time)}")
    print(f"Elapsed: {str(elapsed)}")


# ------------------------------- CLI ---------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Grid benchmark across trackers × OD × ReID (with progress + timing summary)")
    p.add_argument("--tracker", default=None,
                   help=f"One of {SUPPORTED_TRACKERS}; if omitted, runs all.")
    p.add_argument("--od-model", default=None,
                   help=f"One of {SUPPORTED_OD}; if omitted, runs all.")
    p.add_argument("--reid-model", default=None,
                   help=f"One of {SUPPORTED_REID}; if omitted and tracker uses ReID, runs all.")
    p.add_argument("--weight", default=None, help="Optional OD checkpoint (e.g., YOLO .pt)")
    p.add_argument("--coco-root", required=True, help="Path to COCO 2017 root")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--conf-thres", type=float, default=0.25)
    p.add_argument("--images-per-bin", type=int, default=3,
                   help="Max images sampled per object-count bin")
    p.add_argument("--repeats", type=int, default=8,
                   help="timed update() calls per image")
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--compile-od", action="store_true", default=False)
    p.add_argument("--compile-reid", action="store_true",default=False)
    p.add_argument("--start", type=int, default=10, help="Start of object-count range (inclusive)")
    p.add_argument("--end", type=int, default=10, help="End of object-count range (inclusive)")
    p.add_argument("--step", type=int, default=1, help="Step for object-count range")
    p.add_argument("--csv-out", default=None, help="CSV output filename (default: benchmark_output_TIMESTAMP.csv)")
    p.add_argument("--tracker-kwargs", default="{}", help='JSON dict, e.g. \'{"match_iou":0.3}\'')

    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()

    # normalize selections
    trackers = [a.tracker] if a.tracker else SUPPORTED_TRACKERS
    od_models = [a.od_model] if a.od_model else SUPPORTED_OD
    reid_models = [a.reid_model] if a.reid_model else SUPPORTED_REID

    # basic validation hints
    for t in trackers:
        if t not in SUPPORTED_TRACKERS:
            raise ValueError(f"Unsupported tracker: {t}. Supported: {SUPPORTED_TRACKERS}")
    for m in od_models:
        if m not in SUPPORTED_OD:
            raise ValueError(f"Unsupported OD model: {m}. Supported: {SUPPORTED_OD}")
    if a.reid_model and a.reid_model not in SUPPORTED_REID:
        raise ValueError(f"Unsupported ReID model: {a.reid_model}. Supported: {SUPPORTED_REID}")

    run_benchmark_grid(
        trackers=trackers,
        od_models=od_models,
        reid_models=reid_models,
        weight=a.weight,
        coco_root=a.coco_root,
        device_str=a.device,
        conf_thres=a.conf_thres,
        images_per_bin=a.images_per_bin,
        repeats=a.repeats,
        warmup_iters=a.warmup_iters,
        compile_od=a.compile_od,
        compile_reid=a.compile_reid,
        start_bin=a.start,
        end_bin=a.end,
        step_bin=a.step,
        csv_out=a.csv_out,
        tracker_kwargs_json=a.tracker_kwargs,
    )