"""
multi_camera_tracker.py – Cross-camera ID association with mutual-nearest-neighbor matching.

v1.2 changes:
  • Re-normalise features before similarity.
  • Ignore tracks with `hits < min_hits`.
  • Use **mutual best match** criterion to avoid collapsing all IDs.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

from object_tracker.single_camera_tracker import SingleCameraTracker

__all__ = ["MultiCameraTracker"]


class MultiCameraTracker:
    """Associate objects across multiple camera views using appearance features."""

    def __init__(
        self,
        trackers: Sequence[SingleCameraTracker],
        *,
        cross_thr: float = 0.8,
        min_hits: int = 2,
    ) -> None:
        if not trackers:
            raise ValueError("trackers must not be empty")
        self.trackers: List[SingleCameraTracker] = list(trackers)
        self.cross_thr = cross_thr
        self.min_hits = min_hits

        self._id_map: Dict[Tuple[int, int], int] = {}
        self._gid_gen = itertools.count()

    # ------------------------------------------------------------------ #
    #                            helpers                                  #
    # ------------------------------------------------------------------ #

    def _ensure_global(self, key: Tuple[int, int]) -> int:
        if key not in self._id_map:
            self._id_map[key] = next(self._gid_gen)
        return self._id_map[key]

    def _merge_ids(self, a: int, b: int) -> None:
        if a == b:
            return
        hi, lo = max(a, b), min(a, b)
        for k in list(self._id_map):
            if self._id_map[k] == hi:
                self._id_map[k] = lo

    # ------------------------------------------------------------------ #
    #                              update                                 #
    # ------------------------------------------------------------------ #

    def update(self, frames_bgr: Sequence[np.ndarray]) -> List[List[List[float]]]:
        if len(frames_bgr) != len(self.trackers):
            raise ValueError("frames/trackers length mismatch")

        cams_tracks: List[List[List[float]]] = []
        orig_key_by_row: Dict[Tuple[int, int], Tuple[int, int]] = {}

        all_feats: List[np.ndarray] = []
        all_keys: List[Tuple[int, int]] = []

        # 1) Per-camera tracking & feature collection
        for cam_idx, (tracker, frame) in enumerate(zip(self.trackers, frames_bgr)):
            local_tracks = tracker.update(frame)  # [x1,y1,x2,y2,lid,hits]
            cam_out: List[List[float]] = []

            feat_lookup = {t.id: t.feat.detach().cpu().numpy() for t in tracker.tracker.tracks}
            hits_lookup = {t.id: t.hits for t in tracker.tracker.tracks}

            for row_idx, (x1, y1, x2, y2, loc_id, hits) in enumerate(local_tracks):
                key = (cam_idx, int(loc_id))
                gid = self._ensure_global(key)
                cam_out.append([x1, y1, x2, y2, float(gid), hits])
                orig_key_by_row[(cam_idx, row_idx)] = key

                if loc_id in feat_lookup and hits_lookup[loc_id] >= self.min_hits:
                    f = feat_lookup[loc_id]
                    f = f / (np.linalg.norm(f) + 1e-8)  # ensure L2 = 1
                    all_feats.append(f.astype(np.float32))
                    all_keys.append(key)

            cams_tracks.append(cam_out)

        # 2) Mutual-nearest-neighbor cross-camera association
        if len(all_feats) >= 2:
            feats = np.stack(all_feats)            # (N,D)
            sims = feats @ feats.T                 # cosine similarity
            np.fill_diagonal(sims, -1.0)

            # best match across other cams only
            best = sims.argmax(axis=1)
            for i, j in enumerate(best):
                if i >= j:  # handle each pair once
                    continue
                if best[j] != i:
                    continue                     # not mutual
                if all_keys[i][0] == all_keys[j][0]:
                    continue                     # same camera
                if sims[i, j] < self.cross_thr:
                    continue
                self._merge_ids(self._id_map[all_keys[i]], self._id_map[all_keys[j]])

        # 3) Rewrite outputs with final GIDs
        for (cam_idx, row_idx), key in orig_key_by_row.items():
            cams_tracks[cam_idx][row_idx][4] = float(self._id_map[key])

        return cams_tracks

    # ------------------------------------------------------------------ #
    #                           visual helper                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def draw(frames: Sequence[np.ndarray], cams_tracks: Sequence[Sequence[Sequence[float]]]):
        for frame, tracks in zip(frames, cams_tracks):
            SingleCameraTracker.draw(frame, tracks, show_ids=True)
