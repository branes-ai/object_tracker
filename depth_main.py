#!/usr/bin/env python3
"""
demo_depth_camera.py
Monocular depth estimation from a video file or webcam.

Examples
--------
# Process full video
python demo_depth_camera.py --source path/to/video.mp4 --out out_depth.mp4 --side-by-side

# Process only a 30 sec clip starting at 1 min
python demo_depth_camera.py --source path/to/video.mp4 --start 60 --duration 30 --out out.mp4

# Live webcam (index 0)
python demo_depth_camera.py --source 0 --overlay
"""

from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Adjust to your project layout if needed
from branes_platform.nn.depth.models import DepthModel
from branes_platform.applications.distance.visualize_depth import depth_to_color


def open_source(src: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(int(src) if str(src).isdigit() else src)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monocular depth camera demo")
    p.add_argument("--source", required=True,
                   help="video path or webcam index (e.g., 0)")
    p.add_argument("--device", default="cpu", help="torch device (cpu|cuda:0)")
    p.add_argument("--weight", default="Intel/dpt-hybrid-midas",
                   help="HF checkpoint (e.g., Intel/dpt-hybrid-midas, Intel/dpt-swinv2-tiny-256)")
    p.add_argument("--out", default=None, help="optional output video path (mp4)")
    p.add_argument("--start", type=float, default=0.0,
                   help="start time in seconds (video only)")
    p.add_argument("--duration", type=float, default=None,
                   help="clip duration in seconds (video only)")
    view = p.add_mutually_exclusive_group()
    view.add_argument("--overlay", action="store_true", help="overlay depth on RGB")
    view.add_argument("--side-by-side", action="store_true", help="show RGB | depth")
    p.add_argument("--alpha", type=float, default=0.5, help="overlay opacity (when --overlay)")
    p.add_argument("--smooth-alpha", type=float, default=0.0,
                   help="temporal EMA for depth map in [0..1] (0 disables)")
    p.add_argument("--log-interval", type=float, default=1.0, help="seconds between FPS logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    log = logging.getLogger("depth-cam")

    cap = open_source(args.source)
    if not cap.isOpened():
        log.error("Could not open source %s", args.source)
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Seek to start time (if video)
    if not str(args.source).isdigit() and args.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)

    # Output writer
    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(Path(args.out)),
            fourcc,
            fps_src,
            (w if not args.side_by_side else 2 * w, h),
        )

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    depth_model = DepthModel(device=device, weight=args.weight)

    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Stats
    frame_cnt = 0
    t0 = last_log = time.perf_counter()

    # Temporal smoothing
    dnorm_ema = None
    alpha_s = float(args.smooth_alpha)
    use_overlay = bool(args.overlay) or not args.side_by_side

    # Stop condition for duration
    max_frames = None
    if args.duration is not None:
        max_frames = int(args.duration * fps_src)

    while True:
        if max_frames is not None and frame_cnt >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Predict depth
        res = depth_model.predict(frame)
        dnorm = res.depth_norm

        if alpha_s > 0.0:
            if dnorm_ema is None:
                dnorm_ema = dnorm.clone()
            else:
                dnorm_ema = (1 - alpha_s) * dnorm_ema + alpha_s * dnorm
            dshow = dnorm_ema
        else:
            dshow = dnorm

        depth_color = depth_to_color(dshow)

        if use_overlay:
            vis = frame.copy()
            cv2.addWeighted(depth_color, args.alpha, vis, 1 - args.alpha, 0, vis)
        else:
            vis = np.concatenate([frame, depth_color], axis=1)

        frame_cnt += 1
        now = time.perf_counter()
        if now - last_log >= args.log_interval:
            fps = frame_cnt / (now - t0)
            log.info("FPS: %.1f | frame: %dx%d | model=%s", fps, frame.shape[1], frame.shape[0], args.weight)
            last_log = now

        cv2.imshow("Depth", vis)
        if writer is not None:
            writer.write(vis)

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()