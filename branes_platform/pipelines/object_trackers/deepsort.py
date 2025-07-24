"""
tracker.py – DeepSort tracker refactored to use the unified wrappers.

Author  : You
Created : 2025‑06‑25

Usage example
-------------
>>> od   = ODModel("yolo")
>>> reid = ReIDModel("clip")
>>> sort = DeepSort(reid)
>>> dets = od.predict(frame)               # (N,6) tensor [x1 y1 x2 y2 conf cls]
>>> tracks = sort.update(frame, dets)
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Union
from itertools import count

import numpy as np
import torch
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


__all__ = [
    "DeepSort",
]

from branes_platform.nn.reid.reid import ReIDModel


# --------------------------------------------------------------------------- #
#                         helpers – IoU, NMS, sanity                           #
# --------------------------------------------------------------------------- #

def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # (N,4) vs (M,4)
    """Vectorised intersection‑over‑union."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (
        ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
        + ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :] - inter + 1e-7
    )
    return iou


def _nms(tracks: List["_Track"], iou_thr: float = 0.7) -> List["_Track"]:
    if len(tracks) <= 1:
        return tracks
    boxes = np.stack([t.to_xyxy() for t in tracks])
    scores = np.array([t.hits for t in tracks], dtype=np.float32)  # proxy conf
    idxs = scores.argsort()[::-1]

    keep: list[_Track] = []
    while idxs.size:
        i = idxs[0]
        keep.append(tracks[i])
        if idxs.size == 1:
            break
        rest = idxs[1:]
        iou = _iou(boxes[[i]], boxes[rest])[0]
        idxs = rest[iou < iou_thr]
    return keep


def _valid_box(box: np.ndarray, min_size: int = 10) -> bool:
    if np.isnan(box).any():
        return False
    w, h = box[2] - box[0], box[3] - box[1]
    return w >= min_size and h >= min_size

# --------------------------------------------------------------------------- #
#                               single track                                  #
# --------------------------------------------------------------------------- #


class _Track:
    """Internal Kalman‑based track."""

    _ids = count()

    def __init__(self, xyxy: np.ndarray, feat: torch.Tensor):
        self.id: int = next(self._ids)
        self.kf = self._init_kf(xyxy)
        self.time: int = 0        # #frames since last update
        self.hits: int = 1
        self.age: int = 0
        self.confirmed: bool = False
        self.feat: torch.Tensor = feat.clone()

    # --------------------------------------------------------------------- #
    #                              Kalman filter                            #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _xyxy_to_state(b: Sequence[float]) -> Tuple[float, float, float, float]:
        w, h = b[2] - b[0], b[3] - b[1]
        return b[0] + w / 2, b[1] + h / 2, w * h, w / (h + 1e-6)

    def _init_kf(self, xyxy: np.ndarray) -> KalmanFilter:
        cx, cy, s, r = self._xyxy_to_state(xyxy)
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.eye(7, dtype=np.float32)
        kf.F[0, 4] = kf.F[1, 5] = kf.F[2, 6] = 1.0  # constant velocity
        kf.H = np.eye(4, 7, dtype=np.float32)
        kf.R *= 10.0
        kf.P *= 10.0
        kf.x[:4] = np.array([cx, cy, s, r], dtype=np.float32).reshape(-1, 1)
        return kf

    # ------------------------------------------------------------------ #

    def predict(self) -> None:
        self.kf.predict()
        self.time += 1
        self.age += 1

    def update(self, xyxy: np.ndarray, feat: torch.Tensor) -> None:
        self.time = 0
        self.hits += 1
        if self.hits >= 3:
            self.confirmed = True
        self.feat = 0.9 * self.feat + 0.1 * feat
        z = np.array(self._xyxy_to_state(xyxy), dtype=np.float32).reshape(-1, 1)
        self.kf.update(z)

    # ------------------------------------------------------------------ #

    def to_xyxy(self) -> np.ndarray:
        cx, cy, s, r = self.kf.x[:4].flatten()
        if s <= 0 or r <= 0 or not np.isfinite([cx, cy, s, r]).all():
            return np.zeros(4, dtype=np.float32)
        w, h = np.sqrt(s * r), np.sqrt(s / r)
        if w < 10 or h < 10:
            return np.zeros(4, dtype=np.float32)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)

# --------------------------------------------------------------------------- #
#                                 DeepSort                                    #
# --------------------------------------------------------------------------- #


class DeepSort:
    """Classic DeepSort tracker working with unified OD/ReID wrappers.

    Parameters
    ----------
    reid : ReIDModel
        Appearance encoder that produces (D,) features per crop.
    max_age : int, default 30
        Max #frames to keep lost tracks alive.
    iou_thres : float, default 0.4
        IoU threshold for matching.
    appearance_thres : float, default 0.5
        Cosine‑similarity gate (0‑1), lower is stricter.
    """

    def __init__(
        self,
        reid: ReIDModel,
        *,
        max_age: int = 30,
        iou_thres: float = 0.4,
        appearance_thres: float = 0.5,
    ) -> None:
        self.reid = reid
        self.tracks: list[_Track] = []
        self.max_age = max_age
        self.iou_thr = iou_thres
        self.app_thr = appearance_thres

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def update(
        self,
        frame_bgr: np.ndarray,
        detections: Union[torch.Tensor, np.ndarray],
    ) -> List[List[float]]:
        """Update tracker with detections from the current frame.

        Parameters
        ----------
        frame_bgr : ndarray (H,W,3) uint8
            Current frame in **BGR** format.
        detections : (N,6) tensor / ndarray
            [x1,y1,x2,y2,conf,label] coordinates **in pixels**.

        Returns
        -------
        List[[x1,y1,x2,y2,track_id,hits]] – confirmed tracks this frame.
        """
        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()
        if detections.size == 0:
            detections = detections.reshape(0, 6)

        # 1) predict state for existing tracks
        for t in self.tracks:
            t.predict()

        # 2) compute appearance embeddings for detections
        boxes_xyxy = torch.as_tensor(detections[:, :4], dtype=torch.float32, device=self.reid.device)
        feats = self.reid.predict(frame_bgr, boxes_xyxy)

        # 3) build cost matrix (combined IoU + cosine)
        matches: list[tuple[int, int]] = []
        if self.tracks and detections.size:
            iou_mat = _iou(
                np.stack([t.to_xyxy() for t in self.tracks]),
                detections[:, :4].astype(np.float32),
            )
            feats_trk = torch.stack([t.feat for t in self.tracks]) if self.tracks else torch.empty((0, feats.shape[1]))
            app_cos = 1.0 - torch.cdist(feats_trk, feats, p=2).cpu().numpy()  # cosine sim in [‑1,1]

            # combine
            use_iou = (iou_mat > 0.10).astype(np.float32)
            cost = 1.0 - (use_iou * 0.3 * iou_mat + 0.7 * app_cos)
            if not np.isfinite(cost).all():
                cost = np.full_like(cost, 1.0)

            row_ind, col_ind = linear_sum_assignment(cost)
            matches = [
                (r, c)
                for r, c in zip(row_ind, col_ind)
                if cost[r, c] < (1.0 - self.app_thr)
            ]

        # 4) mark matched / unmatched
        matched_trks = {r for r, _ in matches}
        matched_dets = {c for _, c in matches}

        # 5) update tracks
        for r, c in matches:
            self.tracks[r].update(detections[c, :4].astype(np.float32), feats[c])

        # 6) create new tracks for unmatched detections over conf gate
        for idx, det in enumerate(detections):
            if idx in matched_dets or det[4] <= 0.5:
                continue
            self.tracks.append(_Track(det[:4].astype(np.float32), feats[idx]))

        # 7) age + prune
        self.tracks = [
            t for t in self.tracks if t.time <= self.max_age and _valid_box(t.to_xyxy())
        ]
        self.tracks = _nms(self.tracks, iou_thr=0.7)

        # 8) produce outputs – confirmed & just updated
        outputs: list[list[float]] = []
        for t in self.tracks:
            if not t.confirmed or t.time > 0:
                continue
            x1, y1, x2, y2 = t.to_xyxy()
            outputs.append([x1, y1, x2, y2, float(t.id), float(t.hits)])
        return outputs
