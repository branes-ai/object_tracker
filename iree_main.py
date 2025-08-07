import cv2
import numpy as np

from branes_platform.pipelines.object_trackers.deepsort import SingleCameraTrackerIREE

def main():
    sct = SingleCameraTrackerIREE()
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read();  assert ok
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tracks = sct.update(frame)
        # sct.draw(frame, tracks)  # reuse your previous draw util
        cv2.imshow("IREE-DeepSORT", frame)
        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()