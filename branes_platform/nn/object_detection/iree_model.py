# od_model_iree.py
from typing import Any, Sequence
import cv2, numpy as np, torch
from ultralytics.utils.ops import non_max_suppression, scale_coords
from branes_platform.nn.base import BranesModel
from branes_platform.utils.iree import load_vmfb, numpy_to_buffer

class ODModelIREE(BranesModel):
    """
    IREE-compiled YOLO-v8n wrapper (static 640×640).
    Exposes the *same* .predict() signature as your torch ODModel.
    """

    def __init__(self, vmfb_path: str = "yolov8n.vmfb", device: str | None = "cpu") -> None:
        super().__init__(device)
        self.mod = load_vmfb(vmfb_path)
        self.infer = self.mod["main_graph"]

        self.img_size = 640  # compile-time constant
        self.stride = 32     # yolov8 stride for scaling boxes

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray, conf_thres: float = 0.3,
                classes: Sequence[int] | None = None) -> torch.Tensor:
        h0, w0 = frame_bgr.shape[:2]

        # Letterbox-resize to 640
        img = cv2.resize(frame_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = img[:, :, ::-1].astype(np.float32) / 255.0      # to RGB, 0-1
        img = np.transpose(img, (2, 0, 1))[None]              # 1×3×H×W

        # IREE inference (returns list of ndarrays)
        preds = self.infer(img)
        raw = np.array(preds)   # (B, n_anchors, 85)

        # NMS using Ultralytics helper (expects torch)
        det = non_max_suppression(torch.from_numpy(raw), conf_thres, 0.45, classes=classes, max_det=300)[0]
        if det is None or len(det) == 0:
            return torch.empty((0, 6), dtype=torch.float32)

        # Rescale boxes back to input image
        det[:, :4] = scale_coords((self.img_size, self.img_size), det[:, :4], (h0, w0)).round()
        return det.to(torch.float32)