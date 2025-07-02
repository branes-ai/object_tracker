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


def _make_grid(frames: list[np.ndarray], tile_w: int) -> np.ndarray:
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
    rows_list = [np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)]
    return np.vstack(rows_list)


# --------------------------------------------------------------------------- #
#                                   CLI                                       #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Multi-camera DeepSORT++ video demo")
    p.add_argument("--sources", required=True, help="Comma-separated video file paths")
    p.add_argument("--od", default="yolo", choices=["yolo", "detr"], help="Detector model")
    p.add_argument("--display-size", type=int, default=640, help="Width of each tile in the grid")
    p.add_argument("--save-grid", default=None, help="Optional output video path")
    p.add_argument("--time-limit", type=int, default=30, help="Limit processing to N seconds")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#                                 main loop                                   #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    src_list = [s.strip() for s in args.sources.split(",") if s.strip()]
    caps = [_open_source(s) for s in src_list]
    if any(not c.isOpened() for c in caps):
        bad = [src for src, cap in zip(src_list, caps) if not cap.isOpened()]
        raise SystemExit(f"Could not open: {', '.join(bad)}")

    fps_list = []
    frame_limit = []
    for cap in caps:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fps_list.append(fps)
        frame_limit.append(int(fps * args.time_limit))  # limit per file

    trackers = [SingleCameraTracker(od_name=args.od) for _ in caps]
    mct = MultiCameraTracker(trackers, cross_thr=0.78)  # tune threshold

    writer = None
    if args.save_grid:
        ok, frame = caps[0].read()
        if not ok:
            raise SystemExit("Failed to grab first frame – check sources")
        tile_w = args.display_size
        tile_h = int(frame.shape[0] / frame.shape[1] * tile_w)
        cols = int(np.ceil(np.sqrt(len(caps))))
        rows = int(np.ceil(len(caps) / cols))
        grid_w, grid_h = cols * tile_w, rows * tile_h
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(Path(args.save_grid)), fourcc, 30, (grid_w, grid_h))
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_counters = [0] * len(caps)

    # Synchronize all streams to the first available frame
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        frames = []
        stop = False
        for idx, cap in enumerate(caps):
            if frame_counters[idx] >= frame_limit[idx]:
                stop = True
                break

            ok, frame = cap.read()
            if not ok:
                stop = True
                break

            frame_counters[idx] += 1
            frames.append(frame)

        if stop or len(frames) < len(caps):
            break

        cams_tracks = mct.update(frames)
        mct.draw(frames, cams_tracks)

        grid = _make_grid(frames, args.display_size)
        cv2.imshow("DeepSORT++ | Multi-cam", grid)
        writer and writer.write(grid)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    for cap in caps:
        cap.release()
    writer and writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
