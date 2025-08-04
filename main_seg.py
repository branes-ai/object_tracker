#!/usr/bin/env python3
"""
demo_segmentation_camera.py
Run unified segmentation (instance / semantic / panoptic) on a webcam or video.

Examples
--------
# Instance segmentation (YOLO) on default webcam, show masks+boxes
python demo_segmentation_camera.py --task instance --backend yolo --source 0

# Semantic (SegFormer ADE20K) on a video file, save output
python demo_segmentation_camera.py --task semantic --backend segformer \
    --source path/to/video.mp4 --out out.mp4

# Panoptic (Mask2Former COCO Panoptic), CPU, with logging every 1.5s
python demo_segmentation_camera.py --task panoptic --backend mask2former \
    --source 0 --device cpu --log-interval 1.5
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Adjust these if your modules live elsewhere
from branes_platform.nn.segmentation.models import SegModel
from branes_platform.applications.segmentation.visualize import (
    overlay_instances,
    overlay_semantic,
    overlay_panoptic,
)

# ------------------------------- helpers -------------------------------- #

def open_source(src: str) -> cv2.VideoCapture:
    """Accept webcam index or path."""
    return cv2.VideoCapture(int(src) if str(src).isdigit() else src)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segmentation camera demo")
    p.add_argument("--task", required=True, choices=["instance", "semantic", "panoptic"],
                   help="Segmentation task to run.")
    p.add_argument("--backend", default=None,
                   help="Backend per task (defaults: yolo | segformer | mask2former).")
    p.add_argument("--weight", default=None, help="Optional checkpoint override for the backend.")
    p.add_argument("--source", default="0", help="Webcam index or video path.")
    p.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:0")
    p.add_argument("--conf-thres", type=float, default=0.25,
                   help="Confidence threshold (instance only).")
    p.add_argument("--mask-format", default="rle", choices=["rle", "bitmap", "polygons"],
                   help="Instance/panoptic mask representation used for visualization.")
    p.add_argument("--out", default=None, help="Optional output video path.")
    p.add_argument("--log-interval", type=float, default=1.0,
                   help="Seconds between log lines.")
    p.add_argument("--window", default=None, help="Custom window title.")
    # Quality-of-life toggles
    p.add_argument("--draw-boxes", action="store_true",
                   help="Draw boxes and labels for instance/panoptic.")
    p.add_argument("--alpha", type=float, default=0.5, help="Mask overlay opacity [0..1].")
    return p.parse_args()

# ------------------------------- main ---------------------------------- #

def main() -> None:
    args = parse_args()

    # Logging
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                        level=logging.INFO)
    log = logging.getLogger("seg-cam")

    # Resolve defaults per task
    backend = args.backend or {"instance": "yolo", "semantic": "segformer", "panoptic": "mask2former"}[args.task]
    window_name = args.window or f"Segmentation – {args.task}/{backend}"

    # Try to open source
    cap = open_source(args.source)
    if not cap.isOpened():
        log.error("Could not open source %s", args.source)
        return

    # Get stream properties (fallbacks are fine)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Optional writer
    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(Path(args.out)), fourcc, fps_src, (w, h))

    # Instantiate model
    torch_device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model_kwargs = {"weight": args.weight} if args.weight else {}
    seg = SegModel(task=args.task, backend=backend, device=torch_device, **model_kwargs)

    # Stats
    frame_cnt = 0
    t0 = last_log = time.perf_counter()

    # UI
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Main loop
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Ensure BGR uint8
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Inference
        if args.task == "instance":
            res = seg.predict(frame, conf_thres=args.conf_thres, mask_format=args.mask_format)
            vis = overlay_instances(frame, res, alpha=args.alpha, draw_boxes=args.draw_boxes, draw_labels=args.draw_boxes)
        elif args.task == "semantic":
            res = seg.predict(frame)
            vis = overlay_semantic(frame, res.sem_labels, alpha=args.alpha)
        else:  # panoptic
            res = seg.predict(frame, mask_format=args.mask_format)
            vis = overlay_panoptic(frame, res, alpha=args.alpha, draw_things_boxes=args.draw_boxes, draw_labels=args.draw_boxes)

        # FPS logging
        frame_cnt += 1
        now = time.perf_counter()
        if now - last_log >= args.log_interval:
            fps = frame_cnt / (now - t0)
            log.info("FPS: %.1f | frame: %dx%d", fps, vis.shape[1], vis.shape[0])
            last_log = now

        # Show & write
        cv2.imshow(window_name, vis)
        if writer is not None:
            writer.write(vis)
        if (cv2.waitKey(1) & 0xFF) == 27:  # Esc to quit
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()