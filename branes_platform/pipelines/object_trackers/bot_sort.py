# --- BoT-SORT ---------------------------------------------------------------- #
from itertools import count
from typing import Union, List

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import time

from branes_platform.utils.timer import _Timer
from branes_platform.nn.reid.models import ReIDModel
from branes_platform.pipelines.object_trackers.deepsort import _valid_box, _iou


class _BoTTrack:
    _ids = count()
    def __init__(self, xyxy: np.ndarray, feat: torch.Tensor, ema: float = 0.9):
        self.id: int = next(self._ids)
        self.kf = self._init_kf(xyxy)
        self.time = 0; self.hits = 1; self.age = 0; self.confirmed = False
        self.feat = feat.clone() if feat.numel() else torch.zeros(1)
        self.ema = ema
        self.last_box: np.ndarray = xyxy.astype(np.float32)

    @staticmethod
    def _xyxy_to_state(b):
        w,h = b[2]-b[0], b[3]-b[1]
        return b[0]+w/2, b[1]+h/2, w*h, w/(h+1e-6)

    def _init_kf(self, xyxy: np.ndarray):
        from filterpy.kalman import KalmanFilter
        cx,cy,s,r = self._xyxy_to_state(xyxy)
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.eye(7, dtype=np.float32); kf.F[0,4]=kf.F[1,5]=kf.F[2,6]=1.0
        kf.H = np.eye(4, 7, dtype=np.float32)
        kf.R *= 10.0; kf.P *= 10.0
        kf.x[:4] = np.array([cx,cy,s,r], dtype=np.float32).reshape(-1,1)
        return kf

    def predict(self):
        self.kf.predict()
        self.time += 1; self.age += 1

    def update(self, xyxy: np.ndarray, feat: torch.Tensor | None):
        self.time = 0; self.hits += 1
        if self.hits >= 3: self.confirmed = True
        if feat is not None and feat.numel():
            self.feat = self.ema * self.feat + (1.0 - self.ema) * feat
        self.last_box = xyxy.astype(np.float32)
        z = np.array(self._xyxy_to_state(self.last_box), dtype=np.float32).reshape(-1,1)
        self.kf.update(z)

    def to_xyxy(self) -> np.ndarray:
        cx, cy, s, r = self.kf.x[:4].flatten()
        if s <= 0 or r <= 0 or not np.isfinite([cx,cy,s,r]).all(): return np.zeros(4, np.float32)
        w, h = np.sqrt(s*r), np.sqrt(s/r)
        if w < 10 or h < 10: return np.zeros(4, np.float32)
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dtype=np.float32)

    def warped_xyxy(self, H: np.ndarray | None) -> np.ndarray:
        """Warp last observed box using homography H (previous->current)."""
        if H is None: return self.to_xyxy()
        x1,y1,x2,y2 = self.last_box
        pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
        pts = cv2.perspectiveTransform(pts[None, :, :], H)[0]
        xys = pts.reshape(-1,2)
        minxy = xys.min(axis=0); maxxy = xys.max(axis=0)
        warped = np.array([minxy[0],minxy[1],maxxy[0],maxxy[1]], dtype=np.float32)
        if not np.isfinite(warped).all(): return self.to_xyxy()
        return warped


class BoTSORT:
    """BoT-SORT: IoU + appearance fusion, optional camera motion compensation.

    Parameters
    ----------
    reid : ReIDModel
    max_age : int
    match_iou : float
    alpha_iou : float       # weight of IoU vs appearance (0..1)
    ema : float             # EMA for track features
    conf_thres : float
    cmc : bool              # enable camera motion compensation
    cmc_min_matches : int   # feature matches required to trust homography
    """

    def __init__(self,
                 reid: ReIDModel,
                 *,
                 max_age: int = 30,
                 match_iou: float = 0.3,
                 alpha_iou: float = 0.6,
                 ema: float = 0.9,
                 conf_thres: float = 0.3,
                 cmc: bool = True,
                 cmc_min_matches: int = 30,
                 timeit:bool = False) -> None:
        self.reid = reid
        self.max_age = max_age
        self.match_iou = match_iou
        self.alpha_iou = float(np.clip(alpha_iou, 0.0, 1.0))
        self.ema = ema
        self.conf_thres = conf_thres
        self.cmc = cmc
        self.cmc_min_matches = cmc_min_matches
        self.tracks: list[_BoTTrack] = []
        self._prev_gray: np.ndarray | None = None
        self._H_prev2curr: np.ndarray | None = None
        self.timeit = timeit

    # -- camera motion compensation -------------------------------------- #
    @staticmethod
    def _estimate_homography(prev_gray: np.ndarray, curr_gray: np.ndarray, min_matches: int) -> np.ndarray | None:
        try:
            orb = cv2.ORB_create(2000)
            kp1, des1 = orb.detectAndCompute(prev_gray, None)
            kp2, des2 = orb.detectAndCompute(curr_gray, None)
            if des1 is None or des2 is None: return None
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            if len(matches) < min_matches: return None
            matches = sorted(matches, key=lambda m: m.distance)[:max(min_matches, len(matches)//2)]
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
            if H is None or mask is None or mask.sum() < min_matches: return None
            return H
        except Exception:
            return None

    # -------------------------------------------------------------------- #
    @torch.no_grad()
    def update(self, frame_bgr: np.ndarray, detections: Union[torch.Tensor, np.ndarray]) -> List[List[float]]:
        reid_ms = 0.0
        cmc_ms = 0.0
        if getattr(self, "timeit", False):
            total_timer = _Timer(torch.device("cpu"))  # tracker logic on CPU
            reid_timer = _Timer(self.reid.device)  # ReID on its device
            cmc_timer = _Timer(torch.device("cpu"))  # ORB/H is CPU
            total_timer.start()
        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()
        if detections.size == 0:
            detections = np.zeros((0,6), np.float32)
        else:
            detections = detections[detections[:,4] >= self.conf_thres]

        # 0) CMC: estimate homography prev->curr
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        H = None
        if self.cmc and self._prev_gray is not None:
            if getattr(self, "timeit", False):
                cmc_timer.start()
            H = self._estimate_homography(self._prev_gray, curr_gray, self.cmc_min_matches)
            if getattr(self, "timeit", False):
                cmc_ms = cmc_timer.stop_ms()
        self._prev_gray = curr_gray

        # 1) predict step
        for t in self.tracks: t.predict()

        # 2) appearance features for detections
        det_boxes = torch.as_tensor(detections[:, :4], dtype=torch.float32, device=self.reid.device)
        # feats_det = self.reid.predict(frame_bgr, det_boxes) if len(detections) else torch.empty((0, self.reid.config.get("embed_dim", 512)), device=self.reid.device)

        if len(detections):
            if getattr(self, "timeit", False):
                reid_timer.start()
            feats_det = self.reid.predict(frame_bgr, det_boxes)
            if getattr(self, "timeit", False):
                reid_ms = reid_timer.stop_ms()
        else:
            feats_det = torch.empty((0, self.reid.config.get("embed_dim", 512)), device=self.reid.device)
        # 3) build fused cost matrix
        matches: list[tuple[int,int]] = []
        if self.tracks and len(detections):
            # IoU: compare warped last boxes (or KF) to current dets
            trk_boxes = np.stack([(t.warped_xyxy(H) if H is not None else t.to_xyxy()) for t in self.tracks])
            iou = _iou(trk_boxes, detections[:, :4].astype(np.float32))

            # Appearance: cosine similarity
            feats_trk = torch.stack([t.feat for t in self.tracks]) if self.tracks else torch.empty((0, feats_det.shape[1]), device=self.reid.device)
            # ensure shapes
            if feats_trk.numel() == 0 or feats_det.numel() == 0:
                app = np.zeros_like(iou)
            else:
                # normalize in case upstream didn’t
                f_t = torch.nn.functional.normalize(feats_trk, dim=1)
                f_d = torch.nn.functional.normalize(feats_det, dim=1)
                app = (f_t @ f_d.t()).clamp(-1, 1).cpu().numpy()  # cosine in [-1,1]

            fused = self.alpha_iou * iou + (1.0 - self.alpha_iou) * ((app + 1.0) / 2.0)  # map cosine to [0,1]
            cost = 1.0 - fused
            cost[fused < self.match_iou] = 1.0

            r_idx, c_idx = linear_sum_assignment(cost)
            for r, c in zip(r_idx, c_idx):
                if fused[r, c] >= self.match_iou:
                    matches.append((r, c))

        matched_trks = {r for r,_ in matches}
        matched_dets = {c for _,c in matches}

        # 4) update matched (EMA features)
        for r, c in matches:
            self.tracks[r].update(detections[c, :4].astype(np.float32), feats_det[c])

        # 5) init new tracks (from remaining high-conf detections)
        for i, det in enumerate(detections):
            if i in matched_dets: continue
            feat_i = feats_det[i] if i < feats_det.shape[0] else torch.zeros(1, device=self.reid.device)
            self.tracks.append(_BoTTrack(det[:4].astype(np.float32), feat_i, ema=self.ema))

        # 6) prune & confirm
        alive = []
        for t in self.tracks:
            if t.hits >= 3: t.confirmed = True
            if t.time <= self.max_age and _valid_box(t.to_xyxy()):
                alive.append(t)
        self.tracks = alive

        # 7) outputs
        out: list[list[float]] = []
        for t in self.tracks:
            if not t.confirmed or t.time > 0: continue
            x1,y1,x2,y2 = t.to_xyxy()
            out.append([x1,y1,x2,y2, float(t.id), float(t.hits)])

        if not getattr(self, "timeit", False):
            return out
        total_ms = total_timer.stop_ms()
        return out, {
                        "reid_ms": float(reid_ms),
                        "cmc_ms": float(cmc_ms),
                        "total_ms": float(total_ms),
            }