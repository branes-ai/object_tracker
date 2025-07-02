"""
multi_camera_tracker.py – Federates several `SingleCameraTracker` instances,
assigns **global IDs** and now performs **cross‑camera matching** using
appearance embeddings so that the *same physical object* seen in overlapping
views keeps a single ID.

Author  : You
Updated : 2025‑06‑25 – added cross‑camera ReID merging
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

from object_tracker.single_camera_tracker import SingleCameraTracker

__all__ = ["MultiCameraTracker"]


class MultiCameraTracker:
    """Track objects in multiple streams **and** merge identities across cameras."""

    def __init__(self, trackers: Sequence[SingleCameraTracker], *, cross_thr: float = 0.7):
        if not trackers:
            raise ValueError("`trackers` list must not be empty")
        self.trackers: List[SingleCameraTracker] = list(trackers)
        self.cross_thr = cross_thr

        # (cam_idx, local_id)  ->  global_id
        self._id_map: Dict[Tuple[int, int], int] = {}
        self._gid_gen = itertools.count()

    # ------------------------------------------------------------------ #

    def _ensure_global(self, key: Tuple[int, int]) -> int:
        if key not in self._id_map:
            self._id_map[key] = next(self._gid_gen)
        return self._id_map[key]

    def _merge_global_ids(self, old_id: int, new_id: int):
        if old_id == new_id:
            return
        hi, lo = max(old_id, new_id), min(old_id, new_id)
        for k in list(self._id_map):
            if self._id_map[k] == hi:
                self._id_map[k] = lo

    # ------------------------------------------------------------------ #

    def update(self, frames_bgr: Sequence[np.ndarray]) -> List[List[List[float]]]:
        if len(frames_bgr) != len(self.trackers):
            raise ValueError("frames length does not match trackers")

        cams_tracks: List[List[List[float]]] = []
        row_key_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        # maps (cam_idx, row_idx) -> (cam_idx, local_id)

        all_feats: List[np.ndarray] = []
        all_keys: List[Tuple[int, int]] = []

        for cam_idx, (tracker, frame) in enumerate(zip(self.trackers, frames_bgr)):
            local_tracks = tracker.update(frame)
            cam_out: List[List[float]] = []

            feat_lookup = {t.id: t.feat.clone().detach().cpu().numpy() for t in tracker.tracker.tracks}

            for row_idx, (x1, y1, x2, y2, loc_id, hits) in enumerate(local_tracks):
                key = (cam_idx, int(loc_id))
                gid = self._ensure_global(key)
                cam_out.append([x1, y1, x2, y2, float(gid), hits])
                row_key_map[(cam_idx, row_idx)] = key

                if loc_id in feat_lookup:
                    all_feats.append(feat_lookup[loc_id])
                    all_keys.append(key)

            cams_tracks.append(cam_out)

        if all_feats:
            feats = np.stack(all_feats)
            sims = feats @ feats.T
            np.fill_diagonal(sims, 0.0)
            N = len(all_keys)
            for i in range(N):
                for j in range(i + 1, N):
                    if all_keys[i][0] == all_keys[j][0]:
                        continue
                    if sims[i, j] > self.cross_thr:
                        self._merge_global_ids(self._id_map[all_keys[i]], self._id_map[all_keys[j]])

        # rewrite outputs with updated gid
        for (cam_idx, row_idx), key in row_key_map.items():
            cams_tracks[cam_idx][row_idx][4] = float(self._id_map[key])

        return cams_tracks

    # ------------------------------------------------------------------ #

    def draw(self, frames: Sequence[np.ndarray], cams_tracks: Sequence[Sequence[Sequence[float]]]):
        for frame, tracks in zip(frames, cams_tracks):
            SingleCameraTracker.draw(frame, tracks, show_ids=True)

