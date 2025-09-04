#!/usr/bin/env python3
"""
quick_check.py — smoke-test all OD and ReID backends on a single image.

Usage:
  python quick_check.py /path/to/image.jpg --device cpu
  python quick_check.py /path/to/image.jpg --device cuda
"""

from __future__ import annotations
import argparse
import time
from typing import List

import cv2
import numpy as np
import torch

# Your unified wrappers
from branes_platform.nn.object_detection.models import ODModel, get_supported_od_models
from branes_platform.nn.reid.models import ReIDModel, get_supported_reid_models




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=str, help="Path to image (BGR/any readable by OpenCV).")
    ap.add_argument("--device", type=str, default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--topk", type=int, default=5, help="How many boxes to feed to ReID.")
    ap.add_argument("--conf", type=float, default=0.3, help="Min conf for OD outputs.")
    args = ap.parse_args()

    # Load image
    frame_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    assert frame_bgr is not None, f"Could not read image: {args.image}"
    H, W = frame_bgr.shape[:2]
    print(f"[info] Image loaded: {args.image}  (H={H}, W={W})")

    device = torch.device(args.device if args.device != "mps" else "mps")
    print(f"[info] Using device: {device}")

    od_models=get_supported_od_models()
    reid_models=get_supported_reid_models()
    print(f"[info] Will try OD models:   {od_models}")
    print(f"[info] Will try ReID models: {reid_models}")

    # Run each OD model
    od_results = []
    for name in od_models:
        print(f"\n=== OD: {name} ===")
        try:
            od = ODModel(name, device=device, compile_model=False)
            t0 = time.time()
            dets = od.predict(frame_bgr, conf_thres=args.conf)
            torch.cuda.synchronize() if device.type == "cuda" else None
            dt = (time.time() - t0) * 1000
            dets_np = dets.detach().float().cpu().numpy()
            print(f"{name}: detections shape = {tuple(dets_np.shape)}  time = {dt:.1f} ms")
            if dets_np.size:
                # sort by conf desc, keep topK
                order = np.argsort(-dets_np[:, 4])
                dets_np = dets_np[order][: args.topk]
                print(f"{name}: top-{len(dets_np)} boxes (x1 y1 x2 y2 conf cls):")
                for row in dets_np:
                    print("  ", np.array2string(row, precision=1, floatmode="fixed"))
            od_results.append((name, dets_np))
        except Exception as e:
            print(f"[warn] OD '{name}' failed: {e}")

    # Run each ReID model on top-K boxes from each OD
    for reid_name in reid_models:
        print(f"\n=== ReID: {reid_name} ===")
        try:
            reid = ReIDModel(reid_name, device=device, compile_model=False)
            for od_name, dets_np in od_results:
                if dets_np is None or dets_np.shape[0] == 0:
                    print(f"{reid_name} x {od_name}: no detections → skipping.")
                    continue
                boxes = dets_np[:, :4]
                t0 = time.time()
                feats = reid.predict(frame_bgr, boxes)
                torch.cuda.synchronize() if device.type == "cuda" else None
                dt = (time.time() - t0) * 1000
                feats = feats.detach().float().cpu()
                # Report cosine norms just to sanity-check normalisation
                norms = torch.linalg.vector_norm(feats, dim=1)
                print(
                    f"{reid_name} x {od_name}: feats shape = {tuple(feats.shape)}  "
                    f"time = {dt:.1f} ms  |norm-mean| = {float((norms-1).abs().mean()):.3f}"
                )
        except Exception as e:
            print(f"[warn] ReID '{reid_name}' failed: {e}")

    print("\n[done] Quick check complete.")


if __name__ == "__main__":
    main()