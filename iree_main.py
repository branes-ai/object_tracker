import time

import cv2
import numpy as np

from branes_platform.applications.object_trackers.single_camera_tracker import SingleCameraTrackerIREE
import logging


def main():

    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                        level=logging.INFO)
    log = logging.getLogger("single-cam")


    sct = SingleCameraTrackerIREE()
    cap = cv2.VideoCapture(0)

    frame_cnt = 0
    start = last_log = time.perf_counter()

    def _persistent(track):           # hits index = 5
        return track[5] >= 2

    while True:
        ok, frame = cap.read();  assert ok
        tracks = sct.update(frame)
        sct.draw(frame, tracks)  # reuse your previous draw util
        # 🔸 log every N seconds
        frame_cnt += 1
        now = time.perf_counter()
        if now - last_log >= 1.0:
            fps_cur = frame_cnt / (now - start)
            persistent = sum(_persistent(t) for t in tracks)
            log.info("FPS: %.1f | objects: %d (persistent≥2 hits)", fps_cur, persistent)
            last_log = now

        cv2.imshow("DeepSORT", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc
            break





if __name__ == "__main__":
    main()