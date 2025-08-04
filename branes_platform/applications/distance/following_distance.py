from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math
import time

import cv2
import numpy as np
import torch

# Reuse your OD & tracker
from branes_platform.nn.object_detection.models import ODModel
from branes_platform.pipelines.object_trackers.deepsort import DeepSort  # adjust if different
from branes_platform.nn.reid.models import ReIDModel  # your CLIP reid
# If your paths differ, update imports accordingly.
from branes_platform.nn.depth.models import DepthModel, DepthResult
# ----------------------------- Config / Types ----------------------------- #

# COCO "vehicle-like" classes typically used for following distance.
DEFAULT_VEHICLE_CLASSES = ("car", "truck", "bus", "motorbike", "motorcycle")

@dataclass
class LeadEstimate:
    track_id: Optional[int]
    box_xyxy: Optional[Tuple[float, float, float, float]]
    bbox_h: Optional[float]
    ttc: Optional[float]
    state: str  # "OK" | "CAUTION" | "BRAKE"
    info: Dict[str, float]  # extra: dhdt, smoothed_h, fps, ...

# ----------------------------- Ego-lane ROI -------------------------------- #

def default_ego_roi_polygon(w: int, h: int) -> np.ndarray:
    """A simple trapezoid centered in the image acting as an ego-lane proxy."""
    top_y = int(0.55 * h)
    bot_y = h - 1
    half_top = int(0.18 * w)
    half_bot = int(0.32 * w)
    cx = w // 2
    poly = np.array([
        [cx - half_top, top_y],
        [cx + half_top, top_y],
        [cx + half_bot, bot_y],
        [cx - half_bot, bot_y],
    ], dtype=np.int32)
    return poly

def point_in_poly(pt: Tuple[float, float], poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

# ----------------------------- TTC Tracker --------------------------------- #

class _PerTrackDepthFilter:
    def __init__(self, alpha: float = 0.4):
        self.z_smooth: Optional[float] = None
        self.dzdt_ema: Optional[float] = None
        self.last_t: Optional[float] = None
        self.alpha = alpha
        self.alpha_d = 0.5

    def update(self, z: float, t: float) -> Tuple[float, float]:
        if self.z_smooth is None:
            self.z_smooth = z
        else:
            self.z_smooth = (1 - self.alpha) * self.z_smooth + self.alpha * z
        if self.last_t is None:
            self.last_t = t
            dzdt = 0.0
        else:
            dt = max(1e-3, t - self.last_t)
            dzdt = (z - self.z_smooth) / dt
            self.last_t = t
        if self.dzdt_ema is None:
            self.dzdt_ema = dzdt
        else:
            self.dzdt_ema = (1 - self.alpha_d) * self.dzdt_ema + self.alpha_d * dzdt
        return self.z_smooth, self.dzdt_ema

    @staticmethod
    def ttc_from_z_dzdt(z: float, dzdt: float, eps: float = 1e-4) -> Optional[float]:
        # depth_norm increases when nearer (after our normalization).
        # Approaching => dzdt > eps. TTC ~ z / dzdt seconds.
        if dzdt is None or dzdt <= eps:
            return None
        return float(max(0.0, z) / dzdt)

    @staticmethod
    def ttc_from_h_dhdt(h: float, dhdt: float, eps: float = 1e-3) -> Optional[float]:
        # TTC ~ h / dhdt; valid when dhdt > 0 (approaching). If dhdt<=0, TTC is ∞/undefined.
        if dhdt is None or dhdt <= eps or h <= 0:
            return None
        return float(h / dhdt)


class _PerTrackHeightFilter:
    """Keeps smoothed bbox height and computes dh/dt for TTC."""
    def __init__(self, alpha: float = 0.3):
        self.smoothed_h: Optional[float] = None
        self.last_t: Optional[float] = None
        self.alpha = alpha  # EMA for h
        self.dhdt_ema: Optional[float] = None
        self.alpha_d = 0.4  # EMA for derivative

    def update(self, h: float, t: float) -> Tuple[float, float]:
        # Smooth height
        if self.smoothed_h is None:
            self.smoothed_h = h
        else:
            self.smoothed_h = (1 - self.alpha) * self.smoothed_h + self.alpha * h

        # Derivative
        if self.last_t is None:
            self.last_t = t
            dhdt = 0.0
        else:
            dt = max(1e-3, t - self.last_t)
            dhdt = (h - self.smoothed_h) / dt  # use residual to reduce lag
            self.last_t = t

        if self.dhdt_ema is None:
            self.dhdt_ema = dhdt
        else:
            self.dhdt_ema = (1 - self.alpha_d) * self.dhdt_ema + self.alpha_d * dhdt

        return self.smoothed_h, self.dhdt_ema

    @staticmethod
    def ttc_from_h_dhdt(h: float, dhdt: float, eps: float = 1e-3) -> Optional[float]:
        # TTC ~ h / dhdt; valid when dhdt > 0 (approaching). If dhdt<=0, TTC is ∞/undefined.
        if dhdt is None or dhdt <= eps or h <= 0:
            return None
        return float(h / dhdt)

# ------------------------------- Alert FSM --------------------------------- #

class AlertFSM:
    """Three-state alert with hysteresis on TTC."""
    def __init__(self,
                 thr_brake: float = 1.0, thr_caution: float = 2.0,
                 hysteresis: float = 0.3):
        self.state = "OK"
        self.thr_brake = thr_brake
        self.thr_caution = thr_caution
        self.hyst = hysteresis

    def update(self, ttc: Optional[float]) -> str:
        s = self.state
        if ttc is None or math.isinf(ttc):
            # No approaching motion: relax state slowly
            if s == "BRAKE":
                if self._above(self.thr_brake + self.hyst, ttc):
                    s = "CAUTION"
            elif s == "CAUTION":
                s = "OK"
            self.state = s
            return s

        if s == "OK":
            if ttc < self.thr_caution:
                s = "CAUTION"
            if ttc < self.thr_brake:
                s = "BRAKE"
        elif s == "CAUTION":
            if ttc >= self.thr_caution + self.hyst:
                s = "OK"
            if ttc < self.thr_brake:
                s = "BRAKE"
        else:  # BRAKE
            if ttc >= self.thr_brake + self.hyst:
                s = "CAUTION"
        self.state = s
        return s

    @staticmethod
    def _above(thr: float, ttc: Optional[float]) -> bool:
        return (ttc is not None) and (ttc >= thr)

# ------------------------------ Main system -------------------------------- #

class FollowingDistanceSystem:
    """Runs detection + tracking, selects a lead in ego lane, computes TTC, and returns alert state."""
    def __init__(
        self,
        device: str | torch.device = "cpu",
        *,
        od_weight: Optional[str] = None,
        vehicle_classes: Sequence[str] = DEFAULT_VEHICLE_CLASSES,
        roi_poly: Optional[np.ndarray] = None,
        ttc_alpha: float = 0.3,
        fsm_thr_brake: float = 1.0,
        fsm_thr_caution: float = 2.0,
        fsm_hyst: float = 0.3,
            use_depth: bool = True,
            depth_weight: str = "Intel/dpt-hybrid-midas",
            depth_alpha: float = 0.4
    ) -> None:
        self.device = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu")

        # Detector
        self.od = ODModel(model_name="yolo", device=self.device, weight=od_weight)

        # Map class names -> indices if available
        names = getattr(self.od.model, "names", None)
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]
        self.class_names = names or []
        self.vehicle_cls_ids = self._resolve_vehicle_ids(vehicle_classes)

        # Tracker (DeepSort with CLIP reid)
        reid = ReIDModel("clip", device=self.device)
        self.tracker = DeepSort(reid)

        # ROI
        self.roi_poly: Optional[np.ndarray] = roi_poly  # will be set on first frame if None

        # Per-track height filters
        self.height_filters: Dict[int, _PerTrackHeightFilter] = {}

        # Alert logic
        self.fsm = AlertFSM(thr_brake=fsm_thr_brake, thr_caution=fsm_thr_caution, hysteresis=fsm_hyst)

        # Stats
        self.last_time = time.perf_counter()

        self.use_depth = use_depth
        self.depth: Optional[DepthModel] = DepthModel(device=self.device, weight=depth_weight) if use_depth else None
        self.depth_filters: Dict[int, _PerTrackDepthFilter] = {}
        self.depth_alpha = depth_alpha

    # ----------------------------------------------- #
    def _resolve_vehicle_ids(self, vehicle_classes: Sequence[str]) -> List[int]:
        if not self.class_names:
            # YOLO always has names; fallback if missing
            return [2, 3, 5, 7]  # rough COCO ids: car=2, motorcycle=3, bus=5, truck=7 (may vary)
        ids = []
        norm = {n.lower(): i for i, n in enumerate(self.class_names)}
        for k in vehicle_classes:
            k = k.lower()
            if k in norm:
                ids.append(norm[k])
        if not ids:
            ids = list(range(len(self.class_names)))  # fallback: accept all
        return ids

    # ----------------------------------------------- #
    @torch.no_grad()
    def update(self, frame_bgr: np.ndarray) -> LeadEstimate:
        depth_res: Optional[DepthResult] = None
        if self.use_depth and self.depth is not None:
            depth_res = self.depth.predict(frame_bgr)  # depth_norm in [0,1]

        h, w = frame_bgr.shape[:2]
        if self.roi_poly is None:
            self.roi_poly = default_ego_roi_polygon(w, h)

        # 1) Detect vehicles
        dets = self.od.predict(frame_bgr, conf_thres=0.25, classes=self.vehicle_cls_ids)
        # dets: [N,6] [x1,y1,x2,y2,conf,cls]
        # 2) Track
        tracks = self.tracker.update(frame_bgr, dets)  # list[[x1,y1,x2,y2,track_id,hits]]

        # 3) Select lead in ROI (closest = max bbox height among ROI-contained centroids)
        lead_track = None
        lead_h = -1.0
        for x1, y1, x2, y2, tid, hits in tracks:
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            if not point_in_poly((cx, cy), self.roi_poly):
                continue
            hbox = max(1.0, (y2 - y1))
            if hbox > lead_h:
                lead_h = hbox
                lead_track = (int(tid), (float(x1), float(y1), float(x2), float(y2)))

        t_now = time.perf_counter()
        fps = 1.0 / max(1e-6, (t_now - self.last_time))
        self.last_time = t_now

        if lead_track is None:
            self.fsm.update(None)
            return LeadEstimate(...)

        tid, box = lead_track
        x1, y1, x2, y2 = [int(v) for v in box]
        hbox = max(1.0, (y2 - y1))

        # --- TTC from bbox height ---
        filt_h = self.height_filters.get(tid)
        if filt_h is None:
            filt_h = self.height_filters[tid] = _PerTrackHeightFilter(alpha=0.35)
        sm_h, dhdt = filt_h.update(hbox, t_now)
        ttc_bbox = _PerTrackHeightFilter.ttc_from_h_dhdt(sm_h, dhdt)

        # --- TTC from depth (median in box) ---
        ttc_depth = None
        z_med = None
        if depth_res is not None:
            dnorm = depth_res.depth_norm
            # guard bounds
            x1c, x2c = max(0, x1), min(dnorm.shape[1] - 1, x2)
            y1c, y2c = max(0, y1), min(dnorm.shape[0] - 1, y2)
            if x2c > x1c and y2c > y1c:
                z_patch = dnorm[y1c:y2c, x1c:x2c]
                # robust depth: median
                z_med = float(torch.median(z_patch).item() if hasattr(z_patch, "median") else np.median(z_patch))
                filt_z = self.depth_filters.get(tid)
                if filt_z is None:
                    filt_z = self.depth_filters[tid] = _PerTrackDepthFilter(alpha=self.depth_alpha)
                z_s, dzdt = filt_z.update(z_med, t_now)
                ttc_depth = _PerTrackDepthFilter.ttc_from_z_dzdt(z_s, dzdt)

        # Fuse TTCs (conservative)
        ttc_candidates = [t for t in (ttc_bbox, ttc_depth) if t is not None and np.isfinite(t)]
        ttc_fused = min(ttc_candidates) if ttc_candidates else None
        state = self.fsm.update(ttc_fused)

        # Cleanup stale filters
        alive_ids = {int(t[4]) for t in tracks}
        for k in list(self.height_filters.keys()):
            if k not in alive_ids: self.height_filters.pop(k, None)
        for k in list(self.depth_filters.keys()):
            if k not in alive_ids: self.depth_filters.pop(k, None)

        return LeadEstimate(
            track_id=tid,
            box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
            bbox_h=hbox,
            ttc=ttc_fused,
            state=state,
            info={
                "sm_h": float(sm_h),
                "dhdt": float(dhdt or 0.0),
                "z_med": float(z_med or 0.0) if z_med is not None else 0.0,
                "ttc_bbox": float(ttc_bbox) if ttc_bbox is not None else float("inf"),
                "ttc_depth": float(ttc_depth) if ttc_depth is not None else float("inf"),
                "fps": fps,
            }
        )

    # ----------------------------------------------- #
    def draw(self, frame_bgr: np.ndarray, est: LeadEstimate, *, show_depth: Optional[np.ndarray]=None) -> None:
        """Overlay ROI, lead bbox, TTC, and state."""
        if show_depth is not None:
            cv2.addWeighted(show_depth, 0.35, frame_bgr, 0.65, 0, frame_bgr)
        h, w = frame_bgr.shape[:2]
        if self.roi_poly is None:
            self.roi_poly = default_ego_roi_polygon(w, h)
        # ROI
        cv2.polylines(frame_bgr, [self.roi_poly], isClosed=True, color=(255, 255, 0), thickness=2)

        # Lead bbox
        if est.box_xyxy is not None:
            x1, y1, x2, y2 = [int(v) for v in est.box_xyxy]
            color = {"OK": (0, 200, 0), "CAUTION": (0, 200, 255), "BRAKE": (0, 0, 255)}[est.state]
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        # Text HUD
        ttc_txt = "∞" if (est.ttc is None) else f"{est.ttc:0.2f}s"
        hud = f"State: {est.state} | TTC: {ttc_txt} | FPS: {est.info.get('fps', 0.0):.1f}"
        cv2.rectangle(frame_bgr, (8, 8), (8 + 360, 36), (0, 0, 0), -1)
        cv2.putText(frame_bgr, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        ttc_bbox = est.info.get("ttc_bbox", float("inf"))
        ttc_depth = est.info.get("ttc_depth", float("inf"))
        hud2 = f"ttc_bbox={ttc_bbox if np.isfinite(ttc_bbox) else 'inf'}  ttc_depth={ttc_depth if np.isfinite(ttc_depth) else 'inf'}"
        cv2.putText(frame_bgr, hud2, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)