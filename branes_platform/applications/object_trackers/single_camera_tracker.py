"""
single_camera_tracker.py – glue class that combines OD, ReID and DeepSort to
track objects from a *single* video source.

Author  : You
Created : 2025-06-25
"""
from __future__ import annotations

import inspect
import time
from typing import Any, List, Sequence, Dict, Callable

import cv2
import numpy as np
import torch

from branes_platform.nn.object_detection.iree_model import ODModelIREE
from branes_platform.nn.object_detection.models import ODModel
from branes_platform.nn.reid.iree_model import ReIDModelIREE
from branes_platform.nn.reid.models import ReIDModel

from branes_platform.pipelines.object_trackers.deepsort import DeepSort, _valid_box
from branes_platform.pipelines.object_trackers.sort import Sort
from branes_platform.pipelines.object_trackers.bot_sort import BoTSORT
from branes_platform.pipelines.object_trackers.oc_sort import OCSort
from branes_platform.pipelines.object_trackers.byte_track import ByteTrack

# -----------------------------------------------------------------------------
# Timer utility: precise on CUDA via Events; perf_counter on CPU
# -----------------------------------------------------------------------------



__all__ = [
    "SingleCameraTracker",
]

from branes_platform.utils.timer import _Timer


def _algo_registry() -> Dict[str, Callable[..., Any]]:
    """Map user-facing names to tracker classes."""
    return {
        "deep_sort": DeepSort,
        "deepsort": DeepSort,
        "sort": Sort,
        "bot_sort": BoTSORT,
        "botsort": BoTSORT,
        "oc_sort": OCSort,
        "ocsort": OCSort,
        "bytetrack": ByteTrack,
        "byte_track": ByteTrack,
    }

_IGNORED_KEYS_WARNED = set()

def _filter_kwargs_for_ctor(ctor: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs that the constructor accepts (by name)."""
    if not kwargs:
        return {}
    sig = inspect.signature(ctor)
    accepted = set(sig.parameters.keys())
    # If ctor has **kwargs, pass everything through
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    ignored = tuple(k for k in kwargs.keys() if k not in filtered)

    if ignored:
        key = (ctor.__name__, tuple(sorted(ignored)))
        if key not in _IGNORED_KEYS_WARNED:
            print(f"[tracker kwargs] Ignoring unsupported args for {ctor.__name__}: {ignored}")
            _IGNORED_KEYS_WARNED.add(key)
    return filtered

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
        sort_algorithm: str = "deep_sort",
        compile_od: bool | dict[str, Any] = False,
        compile_reid: bool | dict[str, Any] = False,
        od_kwargs: dict[str, Any] | None = None,
        reid_kwargs: dict[str, Any] | None = None,
        tracker_kwargs: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
        timeit: bool = False,
    ) -> None:
        self.timeit = bool(timeit)
        # models ------------------------------------------------------------- #
        self.od = ODModel(od_name, compile_model=compile_od,device=device, **(od_kwargs or {}),)
        self.reid = ReIDModel(reid_name, compile_model=compile_reid,device=device)

        # tracker ------------------------------------------------------------ #
        # tracker ------------------------------------------------------------ #
        algo = sort_algorithm.lower()
        registry = _algo_registry()
        if algo not in registry:
            raise ValueError(
                f"Unknown sort_algorithm='{sort_algorithm}'. "
                f"Supported: {sorted(set(registry.keys()))}"
            )

        TrackerCls = registry[algo]

        # Decide if this tracker expects ReID in its constructor
        # (DeepSort & BoT-SORT do; SORT/OC-SORT/ByteTrack typically don't).
        needs_reid = "reid" in inspect.signature(TrackerCls).parameters

        tk = tracker_kwargs or {}
        ctor_kwargs = _filter_kwargs_for_ctor(TrackerCls, tk)

        if needs_reid:
            print(f"Using {TrackerCls.__name__} tracker (with ReID)")
            self.tracker = TrackerCls(self.reid, **ctor_kwargs)
        else:
            print(f"Using {TrackerCls.__name__} tracker")
            self.tracker = TrackerCls(**ctor_kwargs)

    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def update(self, frame_bgr: np.ndarray) -> List[List[float]]:
        """Run detection ➜ DeepSort update. Returns active tracks."""
        # OD timing
        t_od = _Timer(self.od.device)
        t_od.start()
        dets = self.od.predict(frame_bgr)  # (N,6) tensor on model.device
        od_ms = t_od.stop_ms() if self.timeit else 0.0

        # Tracker timing (and ReID inside it)
        trk_times = None
        if self.timeit:
            # If tracker returns timings, use them; otherwise measure externally.
            t_trk = _Timer(self.reid.device)
            t_trk.start()
            out = self.tracker.update(frame_bgr, dets)
            trk_ms = t_trk.stop_ms()

            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
                tracks, trk_times = out
            else:
                tracks, trk_times = out, {"reid_ms": 0.0, "total_ms": trk_ms}
        else:
            tracks = self.tracker.update(frame_bgr, dets)


        if not self.timeit:
            return tracks

            # Normalize timing dictionary
        trk_total = float(trk_times.get("total_ms", 0.0)) if trk_times else 0.0
        reid_ms = float(trk_times.get("reid_ms", 0.0)) if trk_times else 0.0
        other_ms = max(0.0, trk_total - reid_ms)

        timings = {
            "od_ms": float(od_ms),
            "reid_ms": reid_ms,
            "other_ms": other_ms,
            "total_ms": float(od_ms) + trk_total,
        }
        return tracks, timings

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