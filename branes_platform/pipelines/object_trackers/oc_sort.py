# --- OC-SORT ----------------------------------------------------------------- #
from itertools import count
from typing import Union, List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import time

from branes_platform.utils.timer import _Timer
from branes_platform.pipelines.object_trackers.deepsort import _valid_box, _iou


class _OCTrack:
    _ids = count()

    def __init__(self, xyxy: np.ndarray):
        self.id: int = next(self._ids)
        self.kf = self._init_kf(xyxy)
        self.time: int = 0
        self.hits: int = 1
        self.age: int = 0
        self.confirmed: bool = False
        self.obs_prev: np.ndarray | None = None
        self.obs_curr: np.ndarray = xyxy.astype(np.float32)

    @staticmethod
    def _xyxy_to_state(b):
        w, h = b[2] - b[0], b[3] - b[1]
        return b[0] + w / 2, b[1] + h / 2, w * h, w / (h + 1e-6)

    def _init_kf(self, xyxy: np.ndarray):
        from filterpy.kalman import KalmanFilter
        cx, cy, s, r = self._xyxy_to_state(xyxy)
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.eye(7, dtype=np.float32); kf.F[0,4]=kf.F[1,5]=kf.F[2,6]=1.0
        kf.H = np.eye(4, 7, dtype=np.float32)
        kf.R *= 10.0; kf.P *= 10.0
        kf.x[:4] = np.array([cx, cy, s, r], dtype=np.float32).reshape(-1,1)
        return kf

    def predict(self):
        self.kf.predict()
        self.time += 1; self.age += 1

    def update(self, xyxy: np.ndarray):
        self.time = 0; self.hits += 1
        if self.hits >= 3: self.confirmed = True
        self.obs_prev = self.obs_curr
        self.obs_curr = xyxy.astype(np.float32)
        z = np.array(self._xyxy_to_state(self.obs_curr), dtype=np.float32).reshape(-1,1)
        self.kf.update(z)

    def to_xyxy(self) -> np.ndarray:
        cx, cy, s, r = self.kf.x[:4].flatten()
        if s <= 0 or r <= 0 or not np.isfinite([cx,cy,s,r]).all(): return np.zeros(4, np.float32)
        w, h = np.sqrt(s*r), np.sqrt(s/r)
        if w < 10 or h < 10: return np.zeros(4, np.float32)
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dtype=np.float32)

    def obs_extrapolate(self) -> np.ndarray:
        """Obs-centric box prediction using a constant-velocity model in *box corners*."""
        if self.obs_prev is None:  # no velocity yet, fall back to KF box
            return self.to_xyxy()
        v = (self.obs_curr - self.obs_prev)  # (dx1, dy1, dx2, dy2)
        pred = self.obs_curr + v
        # sanity
        if not np.isfinite(pred).all(): return self.to_xyxy()
        return pred.astype(np.float32)


class OCSort:
    """OC-SORT: observation-centric association + KF smoothing. No ReID.

    Parameters
    ----------
    max_age : int
    iou_thres : float       # main IoU gate for matches
    conf_thres : float      # drop very low-confidence dets
    alpha_obs : float       # weight for obs-pred IoU vs KF-pred IoU (blend)
    """

    def __init__(self, *, max_age: int = 30, iou_thres: float = 0.3,
                 conf_thres: float = 0.3, alpha_obs: float = 0.7,timeit: bool = False) -> None:
        self.tracks: list[_OCTrack] = []
        self.max_age = max_age
        self.iou_thr = iou_thres
        self.conf_thres = conf_thres
        self.alpha_obs = float(np.clip(alpha_obs, 0.0, 1.0))
        self.timeit = timeit

    @torch.no_grad()
    def update(self, frame_bgr: np.ndarray, detections: Union[torch.Tensor, np.ndarray]) -> List[List[float]]:
        trk_timer = _Timer(torch.device("cpu"))  # OC-SORT is CPU-only here; change if you move KF to GPU
        if self.timeit:
            trk_timer.start()

        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()
        if detections.size == 0:
            detections = np.zeros((0,6), np.float32)
        else:
            detections = detections[detections[:,4] >= self.conf_thres]

        # 1) predict
        for t in self.tracks: t.predict()

        matches: list[tuple[int,int]] = []
        if self.tracks and len(detections):
            det_boxes = detections[:, :4].astype(np.float32)
            # obs-centric IoU
            obs_preds = np.stack([t.obs_extrapolate() for t in self.tracks])
            iou_obs = _iou(obs_preds, det_boxes)
            # KF-centric IoU
            kf_preds = np.stack([t.to_xyxy() for t in self.tracks])
            iou_kf = _iou(kf_preds, det_boxes)

            iou_mix = self.alpha_obs * iou_obs + (1.0 - self.alpha_obs) * iou_kf
            cost = 1.0 - iou_mix
            cost[iou_mix < self.iou_thr] = 1.0

            r_idx, c_idx = linear_sum_assignment(cost)
            for r, c in zip(r_idx, c_idx):
                if iou_mix[r, c] >= self.iou_thr:
                    matches.append((r, c))

        matched_trks = {r for r,_ in matches}
        matched_dets = {c for _,c in matches}

        # 2) update matched
        for r, c in matches:
            self.tracks[r].update(detections[c, :4].astype(np.float32))

        # 3) init new
        for i, det in enumerate(detections):
            if i in matched_dets: continue
            self.tracks.append(_OCTrack(det[:4].astype(np.float32)))

        # 4) prune
        alive = []
        for t in self.tracks:
            if t.hits >= 3: t.confirmed = True
            if t.time <= self.max_age and _valid_box(t.to_xyxy()):
                alive.append(t)
        self.tracks = alive

        # 5) outputs
        out: list[list[float]] = []
        for t in self.tracks:
            if not t.confirmed or t.time > 0: continue
            x1,y1,x2,y2 = t.to_xyxy()
            out.append([x1,y1,x2,y2, float(t.id), float(t.hits)])
        if not self.timeit:
            return out
        total_ms = trk_timer.stop_ms()
        return out, {"reid_ms": 0.0, "total_ms": float(total_ms)}