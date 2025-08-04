#!/usr/bin/env python3
from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from branes_platform.applications.distance.following_distance import FollowingDistanceSystem

def open_source(src: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(int(src) if str(src).isdigit() else src)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Following distance & collision warning demo")
    p.add_argument("--source", default="0", help="webcam index or video path")
    p.add_argument("--device", default="cpu", help="torch device (cpu|cuda:0)")
    p.add_argument("--out", default=None, help="optional output video path")
    p.add_argument("--log-interval", type=float, default=1.0, help="seconds between log lines")
    p.add_argument("--thr-brake", type=float, default=1.0, help="TTC threshold for BRAKE (s)")
    p.add_argument("--thr-caution", type=float, default=2.0, help="TTC threshold for CAUTION (s)")
    p.add_argument("--hyst", type=float, default=0.3, help="Hysteresis on thresholds (s)")
    p.add_argument("--use-depth", action="store_true", help="Enable monocular depth model.")
    p.add_argument("--depth-weight", default="Intel/dpt-hybrid-midas", help="Depth model checkpoint.")
    p.add_argument("--show-depth", action="store_true", help="Overlay depth colormap.")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    log = logging.getLogger("follow-dist")

    cap = open_source(args.source)
    if not cap.isOpened():
        log.error("Could not open source %s", args.source)
        return

    w, h = int(cap.get(3)) or 640, int(cap.get(4)) or 480
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(Path(args.out)), fourcc, fps_src, (w, h))

    sys = FollowingDistanceSystem(
        device=args.device,
        fsm_thr_brake=args.thr_brake,
        fsm_thr_caution=args.thr_caution,
        fsm_hyst=args.hyst,
        use_depth=args.use_depth,
        depth_weight=args.depth_weight,
    )

    frame_cnt = 0
    start = last_log = time.perf_counter()
    cv2.namedWindow("Following Distance", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        est = sys.update(frame)
        sys.draw(frame, est)

        frame_cnt += 1
        now = time.perf_counter()
        if now - last_log >= args.log_interval:
            fps_cur = frame_cnt / (now - start)
            log.info("FPS: %.1f | state=%s | TTC=%s",
                     fps_cur, est.state, "inf" if est.ttc is None else f"{est.ttc:.2f}s")
            last_log = now

        cv2.imshow("Following Distance", frame)
        writer and writer.write(frame)
        depth_vis = None
        if args.use_depth and args.show_depth and sys.depth is not None:
            # predict once for the frame (we already call within sys.update, but for visualization
            # we call again to avoid changing the sys API; if you prefer, refactor update to return depth)
            dres = sys.depth.predict(frame)
            from branes_platform.applications.distance.visualize_depth import overlay_depth, depth_to_color
            depth_vis = depth_to_color(dres.depth_norm)
        est = sys.update(frame)
        # Draw with optional depth overlay
        sys.draw(frame, est, show_depth=depth_vis if (args.use_depth and args.show_depth) else None)

        if cv2.waitKey(1) & 0xFF == 27:  # Esc
            break

    cap.release()
    writer and writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()