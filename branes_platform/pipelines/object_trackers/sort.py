# --- SORT ------------------------------------------------------------------- #
from typing import List, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from branes_platform.pipelines.object_trackers.deepsort import _Track, _iou, _valid_box


class Sort:
    """Classic SORT: KF + IoU Hungarian, no appearance.

    Params
    ------
    max_age : keep a track alive this many missed frames
    iou_thres : minimum IoU to accept a match
    min_hits : #updates before a track is considered confirmed
    conf_thres : ignore detections below this confidence
    """

    def __init__(self, *, max_age: int = 30, iou_thres: float = 0.3, min_hits: int = 3, conf_thres: float = 0.3) -> None:
        self.tracks: list[_Track] = []
        self.max_age = max_age
        self.iou_thr = iou_thres
        self.min_hits = min_hits
        self.conf_thres = conf_thres

    @torch.no_grad()
    def update(self, frame_bgr: np.ndarray, detections: Union[torch.Tensor, np.ndarray]) -> List[List[float]]:
        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()
        if detections.size == 0:
            detections = np.zeros((0, 6), dtype=np.float32)
        else:
            detections = detections[detections[:, 4] >= self.conf_thres]

        # 1) predict step
        for t in self.tracks:
            t.predict()

        # 2) association with IoU
        matches: list[tuple[int,int]] = []
        if self.tracks and len(detections):
            trk_boxes = np.stack([t.to_xyxy() for t in self.tracks])
            det_boxes = detections[:, :4].astype(np.float32)
            iou = _iou(trk_boxes, det_boxes)
            # convert to cost (Hungarian solves min cost)
            cost = 1.0 - iou
            cost[ iou < self.iou_thr ] = 1.0  # disallow
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if iou[r, c] >= self.iou_thr:
                    matches.append((r, c))

        matched_trks = {r for r, _ in matches}
        matched_dets = {c for _, c in matches}

        # 3) update matched
        for r, c in matches:
            self.tracks[r].update(detections[c, :4].astype(np.float32), feat=torch.zeros(1))  # no appearance

        # 4) init new tracks from unmatched detections
        for idx, det in enumerate(detections):
            if idx in matched_dets:
                continue
            self.tracks.append(_Track(det[:4].astype(np.float32), feat=torch.zeros(1)))

        # 5) age & prune
        alive: list[_Track] = []
        for t in self.tracks:
            if t.hits >= self.min_hits:
                t.confirmed = True
            if t.time <= self.max_age and _valid_box(t.to_xyxy()):
                alive.append(t)
        self.tracks = alive

        # 6) outputs (confirmed and just updated)
        out: list[list[float]] = []
        for t in self.tracks:
            if not t.confirmed or t.time > 0:
                continue
            x1, y1, x2, y2 = t.to_xyxy()
            out.append([x1, y1, x2, y2, float(t.id), float(t.hits)])
        return out