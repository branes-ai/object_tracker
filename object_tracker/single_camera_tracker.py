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

from object_tracker.deep_learning_models import ODModel, ReIDModel
from object_tracker.deepsort  import DeepSort, _valid_box

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
        od_kwargs: dict[str, Any] | None = None,
        reid_kwargs: dict[str, Any] | None = None,
        tracker_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # models ------------------------------------------------------------- #
        self.od = ODModel(od_name, **(od_kwargs or {}))
        self.reid = ReIDModel(reid_name, **(reid_kwargs or {}))
        tracker_kwargs = tracker_kwargs or {}
        self.tracker = DeepSort(self.reid, **tracker_kwargs)

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
