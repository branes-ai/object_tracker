# --- ByteTrack --------------------------------------------------------------- #
from typing import Union, List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from branes_platform.pipelines.object_trackers.deepsort import _valid_box, _Track, _iou


class ByteTrack:
    """Simplified ByteTrack: two-stage matching with high/low score dets.
    No appearance features. Expects good detector scores.

    Params
    ------
    high_thres : float
        Score threshold for 'reliable' detections (used to init new tracks).
    low_thres : float
        Lower bound to keep candidates for recovery matching.
    match_iou : float
        IoU gate for association.
    max_age : int
        Frames to keep unmatched tracks alive.
    min_hits : int
        #updates before track is confirmed.
    """

    def __init__(
        self,
        *,
        high_thres: float = 0.6,
        low_thres: float = 0.1,
        match_iou: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        assert 0.0 <= low_thres <= high_thres <= 1.0
        self.high_thres = high_thres
        self.low_thres = low_thres
        self.match_iou = match_iou
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[_Track] = []

    @staticmethod
    def _assign(tracks: list[_Track], dets_xyxy: np.ndarray, iou_thr: float) -> tuple[list[tuple[int,int]], set[int], set[int]]:
        if not tracks or dets_xyxy.size == 0:
            return [], set(), set()
        trk_boxes = np.stack([t.to_xyxy() for t in tracks])
        iou = _iou(trk_boxes, dets_xyxy.astype(np.float32))
        cost = 1.0 - iou
        cost[iou < iou_thr] = 1.0
        r_idx, c_idx = linear_sum_assignment(cost)
        matches = [(r, c) for r, c in zip(r_idx, c_idx) if iou[r, c] >= iou_thr]
        matched_trks = {r for r, _ in matches}
        matched_dets = {c for _, c in matches}
        return matches, matched_trks, matched_dets

    @torch.no_grad()
    def update(self, frame_bgr: np.ndarray, detections: Union[torch.Tensor, np.ndarray]) -> List[List[float]]:
        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()
        if detections.size == 0:
            detections = np.zeros((0, 6), dtype=np.float32)

        # split detections
        high_mask = detections[:, 4] >= self.high_thres
        low_mask  = (detections[:, 4] >= self.low_thres) & (~high_mask)
        det_high = detections[high_mask]
        det_low  = detections[low_mask]

        # 1) predict
        for t in self.tracks:
            t.predict()

        # 2) match with high-score first
        matches_h, mtrk_h, mdet_h = self._assign(self.tracks, det_high[:, :4] if len(det_high) else np.zeros((0,4)), self.match_iou)
        for r, c in matches_h:
            self.tracks[r].update(det_high[c, :4].astype(np.float32), feat=torch.zeros(1))

        # 3) unmatched tracks try to match low-score dets (recovery)
        un_tracks = [t for i, t in enumerate(self.tracks) if i not in mtrk_h]
        matches_l, mtrk_l, mdet_l = self._assign(un_tracks, det_low[:, :4] if len(det_low) else np.zeros((0,4)), self.match_iou)
        # map back indices of un_tracks to self.tracks
        un_map = [i for i in range(len(self.tracks)) if i not in mtrk_h]
        for r, c in matches_l:
            self.tracks[un_map[r]].update(det_low[c, :4].astype(np.float32), feat=torch.zeros(1))

        # 4) create new tracks from unmatched *high-score* dets only
        unmatched_high = set(range(len(det_high))) - mdet_h
        for idx in unmatched_high:
            self.tracks.append(_Track(det_high[idx, :4].astype(np.float32), feat=torch.zeros(1)))

        # 5) age & confirm & prune
        alive: list[_Track] = []
        for t in self.tracks:
            if t.hits >= self.min_hits:
                t.confirmed = True
            if t.time <= self.max_age and _valid_box(t.to_xyxy()):
                alive.append(t)
        self.tracks = alive

        # 6) outputs (confirmed & updated this frame)
        out: list[list[float]] = []
        for t in self.tracks:
            if not t.confirmed or t.time > 0:
                continue
            x1, y1, x2, y2 = t.to_xyxy()
            out.append([x1, y1, x2, y2, float(t.id), float(t.hits)])
        return out