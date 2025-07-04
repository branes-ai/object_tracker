"""
main.py – multi-camera DeepSORT++ tracking on video files with dynamic time limit and synchronized start.

Run with:
    python main.py --sources file1.mp4,file2.mp4 --time-limit 30

Optional:
    --od yolo|detr           choose detector (default: yolo)
    --save-grid out.mp4      write a tiled output video
    --display-size 640       set width of each camera tile (default: 640)

The script shows a live window with the tiled streams and global IDs.
Esc closes the window.
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import List, Sequence

import cv2
import numpy as np
from pathlib import Path

from object_tracker.single_camera_tracker import SingleCameraTracker
from object_tracker.multi_camera_tracker import MultiCameraTracker


# --------------------------------------------------------------------------- #
#                               utilities                                     #
# --------------------------------------------------------------------------- #
def _open_source(src: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(src)


def _make_grid(frames: List[np.ndarray], tile_w: int) -> np.ndarray:
    if not frames:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    n = len(frames)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    h_ratio = frames[0].shape[0] / frames[0].shape[1]
    tile_h = int(tile_w * h_ratio)
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    tiles = [cv2.resize(f, (tile_w, tile_h)) for f in frames]
    tiles += [blank] * (rows * cols - n)
    rows_img = [np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)]
    return np.vstack(rows_img)

# --------------------------------------------------------------------------- #
#                                   CLI                                       #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Multi-camera DeepSORT++ demo with logging")
    p.add_argument("--sources", required=True, help="Comma‑separated video file paths")
    p.add_argument("--od", default="yolo", choices=["yolo", "detr"], help="Detector model")
    p.add_argument("--display-size", type=int, default=640, help="Tile width in grid")
    p.add_argument("--save-grid", default=None, help="Optional output grid video path")
    p.add_argument("--time-limit", type=int, default=30, help="Limit processing to N seconds")
    p.add_argument("--log-interval", type=float, default=1.0, help="Stats print interval (s)")
    return p.parse_args()

# --------------------------------------------------------------------------- #
#                                logging setup                                #
# --------------------------------------------------------------------------- #

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("tracker-demo")

# --------------------------------------------------------------------------- #
#                                 main                                        #
# --------------------------------------------------------------------------- #

def _count_persistent(tracks: Sequence[Sequence[float]], min_hits: int = 2) -> int:
    """Count tracks whose *hits* >= min_hits (index 5)."""
    return sum(1 for t in tracks if t[5] >= min_hits)


def main():
    args = parse_args()
    src_paths = [s.strip() for s in args.sources.split(",") if s.strip()]
    caps = [_open_source(p) for p in src_paths]
    if any(not c.isOpened() for c in caps):
        bad = [p for p, c in zip(src_paths, caps) if not c.isOpened()]
        logger.error("Could not open: %s", ", ".join(bad))
        return

    fps_list, frame_limit = [], []
    for cap in caps:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fps_list.append(fps)
        frame_limit.append(int(fps * args.time_limit))

    trackers = [SingleCameraTracker(od_name=args.od) for _ in caps]
    mct = MultiCameraTracker(trackers)

    # optional writer
    writer = None
    if args.save_grid:
        ok, fr = caps[0].read();  caps[0].set(cv2.CAP_PROP_POS_FRAMES, 0)
        if ok:
            tile_w = args.display_size
            tile_h = int(fr.shape[0] / fr.shape[1] * tile_w)
            cols = int(np.ceil(np.sqrt(len(caps))))
            rows = int(np.ceil(len(caps) / cols))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(Path(args.save_grid)), fourcc, 30, (cols * tile_w, rows * tile_h))

    frame_counters = [0] * len(caps)
    start = time.perf_counter()
    last_log = start
    frame_total = 0
    persistent_ids_global: set[int] = set()

    # sync all streams
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        frames: List[np.ndarray] = []
        stop = False
        for idx, cap in enumerate(caps):
            if frame_counters[idx] >= frame_limit[idx]:
                stop = True
                break
            ok, frame = cap.read()
            if not ok:
                stop = True
                break
            frames.append(frame)
            frame_counters[idx] += 1
        if stop or len(frames) < len(caps):
            break

        cams_tracks = mct.update(frames)
        ids_per_cam = [set(int(t[4]) for t in trks) for trks in cams_tracks]
        shared_ids = set.intersection(*ids_per_cam) if len(ids_per_cam) > 1 else set()
        cams_tracks = [
            [t for t in trks if int(t[4]) in shared_ids]  # filter per camera
            for trks in cams_tracks
        ]

        mct.draw(frames, cams_tracks)

        # ---- logging --------------------------------------------------- #
        frame_total += 1
        now = time.perf_counter()
        if now - last_log >= args.log_interval:
            fps_cur = frame_total / (now - start)
            per_cam_persistent = [
                _count_persistent(cam_trks) for cam_trks in cams_tracks
            ]
            # union of global IDs present per cam
            ids_per_cam = [set(int(t[4]) for t in cam_trks) for cam_trks in cams_tracks]
            ids_intersection = set.intersection(*ids_per_cam) if len(ids_per_cam) > 1 else set()
            logger.info(
                "FPS: %.1f | persistent objs/cam: %s | cross‑visible: %d",
                fps_cur,
                per_cam_persistent,
                len(ids_intersection),
            )
            last_log = now

        # ---- display --------------------------------------------------- #
        grid = _make_grid(frames, args.display_size)
        cv2.imshow("DeepSORT++ | Multi‑cam", grid)
        writer and writer.write(grid)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # cleanup
    for cap in caps:
        cap.release()
    writer and writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
