"""
single_camera_tracker.py – glue class that combines OD, ReID and DeepSort to
track objects from a *single* video source.

Author  : You
Created : 2025-06-25
"""
from __future__ import annotations

from typing import Any, List, Sequence

import cv2
import numpy as np
import torch

from branes_platform.nn.object_detection.iree_model import ODModelIREE
from branes_platform.nn.object_detection.models import ODModel
from branes_platform.nn.reid.iree_model import ReIDModelIREE
from branes_platform.nn.reid.models import ReIDModel

from branes_platform.pipelines.object_trackers.deepsort import DeepSort, _valid_box

__all__ = [
    "SingleCameraTracker",
]


class SingleCameraTracker:
    """High-level tracker running on a single video feed.

    Examples
    --------
    >>> sct = SingleCameraTracker(od_name="yolo", reid_name="clip")
    >>> cap = cv2.VideoCapture(0)
    >>> while True:
    ...     ok, frame = cap.read();  assert ok
    ...     tracks = sct.update(frame)
    ...     sct.draw(frame, tracks)
    ...     cv2.imshow("SCT", frame)
    """

    def __init__(
        self,
        *,
        od_name: str = "yolo",
        reid_name: str = "clip",
        compile_od: bool | dict[str, Any] = False,
        compile_reid: bool | dict[str, Any] = False,
        od_kwargs: dict[str, Any] | None = None,
        reid_kwargs: dict[str, Any] | None = None,
        tracker_kwargs: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        # models ------------------------------------------------------------- #
        self.od = ODModel(od_name, compile_model=compile_od,device=device, **(od_kwargs or {}),)
        self.reid = ReIDModel(reid_name, compile_model=compile_reid,device=device)

        self.tracker = DeepSort(self.reid, **(tracker_kwargs or {}))

    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def update(self, frame_bgr: np.ndarray) -> List[List[float]]:
        """Run detection ➜ DeepSort update. Returns active tracks."""
        dets = self.od.predict(frame_bgr)  # (N,6) tensor on model.device
        tracks = self.tracker.update(frame_bgr, dets)
        return tracks

    # --------------------------------------------------------------------- #
    #                           visual helpers                               #
    # --------------------------------------------------------------------- #

    @staticmethod
    def draw(
        frame: np.ndarray,
        tracks: Sequence[Sequence[float]],
        *,
        show_ids: bool = True,
        min_box: int = 5,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        """Draw bounding boxes & ids *in-place* on `frame`."""
        for x1, y1, x2, y2, tid, _ in tracks:
            if np.isnan([x1, y1, x2, y2]).any():
                continue
            if (x2 - x1) < min_box or (y2 - y1) < min_box:
                continue
            if not _valid_box([x1, y1, x2, y2]):
                continue
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, color, 2)
            if show_ids:
                cv2.putText(
                    frame,
                    f"ID {int(tid)}",
                    (p1[0], p1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )


class SingleCameraTrackerIREE(SingleCameraTracker):
    """
    Same user API, but backed by IREE-compiled YOLO & CLIP.
    """
    def __init__(self, *,
                 od_vmfb: str = "yolov8n.vmfb",
                 reid_vmfb: str = "clip_vitb32_visual_cpu.vmfb",
                 tracker_kwargs: dict[str,Any] | None = None,
                 device: str | None = "cpu"):
        self.od   = ODModelIREE(od_vmfb, device)
        self.reid = ReIDModelIREE(reid_vmfb, device=device)
        self.tracker = DeepSort(self.reid, **(tracker_kwargs or {}))

    @torch.no_grad()
    def update(self, frame_bgr: np.ndarray) -> List[List[float]]:
        dets = self.od.predict(frame_bgr)
        return self.tracker.update(frame_bgr, dets)

    # optional draw() identical to your previous SCT