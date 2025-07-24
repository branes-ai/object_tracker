import argparse
import logging
import time
from pathlib import Path

import cv2

from branes_platform.applications.object_trackers.single_camera_tracker import SingleCameraTracker


# ────────────────────────────── helpers ────────────────────────────────────── #
def open_source(src: str) -> cv2.VideoCapture:
    """Accept webcam index or path."""
    return cv2.VideoCapture(int(src) if str(src).isdigit() else src)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepSORT++ single-camera demo")
    p.add_argument("--source", default="0", help="webcam index or video path")
    p.add_argument("--out", default=None, help="optional output video path")
    p.add_argument("--log-interval", type=float, default=1.0,
                   help="seconds between log lines")
    return p.parse_args()


# ────────────────────────────── main ──────────────────────────────────────── #
def main():
    args = parse_args()

    # 🔸 simple python logging
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                        level=logging.INFO)
    log = logging.getLogger("single-cam")

    cap = open_source(args.source)
    if not cap.isOpened():
        log.error("Could not open source %s", args.source)
        return

    w, h = int(cap.get(3)), int(cap.get(4))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(Path(args.out)), fourcc, fps_src, (w, h))

    # tracker
    sct = SingleCameraTracker(
        od_name="yolo",
        tracker_kwargs=dict(max_age=50, iou_thres=0.4, appearance_thres=0.5),
        compile_od=True,
        compile_reid=True,
    )

    # 🔸 stats
    frame_cnt = 0
    start = last_log = time.perf_counter()

    def _persistent(track):           # hits index = 5
        return track[5] >= 2

    # loop
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        tracks = sct.update(frame)
        sct.draw(frame, tracks)

        # 🔸 log every N seconds
        frame_cnt += 1
        now = time.perf_counter()
        if now - last_log >= args.log_interval:
            fps_cur = frame_cnt / (now - start)
            persistent = sum(_persistent(t) for t in tracks)
            log.info("FPS: %.1f | objects: %d (persistent≥2 hits)", fps_cur, persistent)
            last_log = now

        cv2.imshow("DeepSORT", frame)
        writer and writer.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc
            break

    cap.release()
    writer and writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()