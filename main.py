import cv2
from object_tracker.single_camera_tracker import SingleCameraTracker

def open_source(src):
    return cv2.VideoCapture(int(src) if str(src).isdigit() else src)

def main(source=0, out=None):
    cap    = open_source(source)
    w, h   = int(cap.get(3)), int(cap.get(4))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = (cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
              if out else None)

    sct = SingleCameraTracker(
        od_name="yolo",             # or 'detr'
        tracker_kwargs=dict(max_age=50, iou_thres=0.4, appearance_thres=0.5),
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        tracks = sct.update(frame)    # ⇨ DeepSort outputs
        sct.draw(frame, tracks)       # annotate frame

        cv2.imshow('DeepSORT', frame)
        if writer: writer.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:   # Esc
            break

    cap.release()
    writer and writer.release()

if __name__ == "__main__":  # demo
    main(0, out="out.mp4")   # webcam -> out.mp4