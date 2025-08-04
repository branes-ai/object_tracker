"""
visualize.py – lightweight utilities to visualize segmentation results.

Author : You
Created: 2025-07-31

Functions
---------
- overlay_instances(frame_bgr, seg_result, alpha=0.5, ...)
- colorize_semantic(sem_labels, ...)
- overlay_semantic(frame_bgr, sem_labels, alpha=0.5, ...)
- overlay_panoptic(frame_bgr, seg_result, alpha=0.5, ...)

Notes
-----
- Input/Output frames are **BGR**, uint8.
- Works without OpenCV; if cv2 is available, boxes/labels render nicer.
- If seg_result contains RLE masks and pycocotools is missing, you’ll get an
  error. In that case, call `SegModel.predict(..., mask_format="bitmap")`
  or install `pycocotools`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# Import SegResult and RLE decoder from the segmentation module
try:
    from branes_platform.nn.segmentation.models import SegResult, rles_to_bitmaps
except Exception:
    # Fall back types if imported standalone (you can ignore this in your repo)
    SegResult = Any  # type: ignore
    def rles_to_bitmaps(rles, out_h: int, out_w: int):  # type: ignore
        raise RuntimeError("rles_to_bitmaps import failed; ensure you're using this inside branes_platform.nn.seg")


# --------------------------------------------------------------------------- #
#                             Color / Palette utils                            #
# --------------------------------------------------------------------------- #

def _hsv_to_bgr(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """Convert HSV in [0,1] to BGR uint8."""
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = int(255 * v * (1.0 - s))
    q = int(255 * v * (1.0 - f * s))
    t = int(255 * v * (1.0 - (1.0 - f) * s))
    v = int(255 * v)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return b, g, r  # BGR


def make_palette(num: int, seed: int = 42, s: float = 0.68, v: float = 0.95) -> np.ndarray:
    """Deterministic, well-spaced color palette (BGR) of shape [num, 3]."""
    rng = np.random.RandomState(seed)
    # Low-discrepancy-ish sequence on hue wheel
    hues = (np.arange(num) * (math.sqrt(5) - 1) * 0.5) % 1.0
    jitter = (rng.rand(num) - 0.5) * 0.05
    hues = (hues + jitter) % 1.0
    cols = np.stack([np.array(_hsv_to_bgr(float(h), s, v), dtype=np.uint8) for h in hues], axis=0)
    return cols  # [num, 3] BGR


def color_for_id(idx: int, seed: int = 0) -> Tuple[int, int, int]:
    """Color for an arbitrary integer id via hashing to HSV."""
    # Simple mix of bits; stable across runs
    x = (idx * 2654435761 + 0x9E3779B9 + seed) & 0xFFFFFFFF
    h = ((x % 360) / 360.0)
    return _hsv_to_bgr(h, 0.65, 0.95)


# --------------------------------------------------------------------------- #
#                             Drawing / blending                               #
# --------------------------------------------------------------------------- #

def _alpha_blend(dst_bgr: np.ndarray, src_bgr: np.ndarray, mask: np.ndarray, alpha: float) -> None:
    """In-place alpha blend `src_bgr` onto `dst_bgr` where `mask` is True.

    Accepts mask as (H,W) or (H,W,1) or (H,W,3) boolean/uint8.
    """
    if alpha <= 0.0:
        return

    if mask.dtype != np.bool_:
        m = mask.astype(bool)
    else:
        m = mask

    # Normalize mask to (H,W) boolean
    if m.ndim == 3:
        if m.shape[2] == 1:
            m = m[..., 0]
        elif m.shape[2] == 3:
            # reduce along channels (any True counts as covered)
            m = np.any(m, axis=2)
        else:
            raise ValueError(f"Unsupported mask shape {m.shape}; expected (H,W), (H,W,1) or (H,W,3).")

    if m.shape != dst_bgr.shape[:2]:
        raise ValueError(f"Mask spatial shape {m.shape} != image shape {dst_bgr.shape[:2]}.")

    if not np.any(m):
        return

    # Compute blended image once, then select with np.where (broadcasts channel dim)
    blended = (dst_bgr.astype(np.float32) * (1.0 - alpha) + src_bgr.astype(np.float32) * alpha).astype(np.uint8)
    dst_bgr[:] = np.where(m[..., None], blended, dst_bgr)

def _draw_box_and_label(img: np.ndarray, xyxy: Sequence[float], label: str, color: Tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1 = max(x1, 0), max(y1, 0)
    if _HAS_CV2:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = 0.5
            (w, h), baseline = cv2.getTextSize(label, font, fs, thickness=1)
            # Background
            cv2.rectangle(img, (x1, y1 - h - baseline), (x1 + w + 2, y1), color, thickness=-1)
            # Text (white or black depending on brightness)
            txt_color = (0, 0, 0) if (0.299*color[2] + 0.587*color[1] + 0.114*color[0]) > 128 else (255, 255, 255)
            cv2.putText(img, label, (x1 + 1, y1 - baseline), font, fs, txt_color, thickness=1, lineType=cv2.LINE_AA)
    else:
        # Minimal fallback: border rectangle only (no text)
        x2 = min(x2, img.shape[1] - 1)
        y2 = min(y2, img.shape[0] - 1)
        img[y1:y1+2, x1:x2] = color
        img[y2-2:y2, x1:x2] = color
        img[y1:y2, x1:x1+2] = color
        img[y1:y2, x2-2:x2] = color


# --------------------------------------------------------------------------- #
#                         Public visualization helpers                         #
# --------------------------------------------------------------------------- #

def overlay_instances(
    frame_bgr: np.ndarray,
    seg: SegResult,
    *,
    alpha: float = 0.5,
    draw_boxes: bool = True,
    draw_labels: bool = True,
    score_fmt: str = "{:.2f}",
    thickness: int = 2,
    palette: Optional[np.ndarray] = None,   # [K, 3] BGR; else inferred from classes or hashed
    seed: int = 0,
) -> np.ndarray:
    """Overlay instance masks on a BGR frame (and optionally draw boxes/labels).

    Parameters
    ----------
    frame_bgr : (H,W,3) uint8
    seg       : SegResult with instance-style fields populated
                (boxes/scores/labels and masks or RLEs/polygons)
    alpha     : mask overlay opacity in [0,1]
    palette   : color palette per class id; if None, auto-generate

    Returns
    -------
    out_bgr   : (H,W,3) uint8 image with overlays
    """
    assert frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3 and frame_bgr.dtype == np.uint8
    H, W = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    N = seg.num_instances()
    if N == 0:
        return out

    # Resolve masks (bitmap path preferred)
    masks_bool: Optional[np.ndarray] = None
    if seg.masks is not None and seg.masks.numel() > 0:
        masks_bool = seg.masks.cpu().numpy().astype(bool)
    elif getattr(seg, "rles", None):
        try:
            masks_bool = rles_to_bitmaps(seg.rles, H, W).cpu().numpy().astype(bool)  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Could not decode RLE masks. Either install `pycocotools` "
                "or call `SegModel.predict(..., mask_format='bitmap')`."
            ) from e
    elif getattr(seg, "polys", None):
        # Rasterize polygons
        if not _HAS_CV2:
            raise RuntimeError("Rasterizing polygons requires OpenCV. Install `opencv-python(-headless)` or request bitmap masks.")
        masks_bool = np.zeros((N, H, W), dtype=bool)
        for i, poly_list in enumerate(seg.polys):
            for cnt in poly_list:
                cv2.fillPoly(masks_bool[i], [cnt.astype(np.int32)], 1)  # type: ignore
    else:
        return out  # nothing to draw

    # Class names and palette (class-based coloring if available)
    class_names: Optional[List[str]] = None
    if isinstance(seg.config, dict):
        names = seg.config.get("classes")
        if isinstance(names, (list, tuple)):
            class_names = list(names)

    labels = seg.labels.cpu().numpy().tolist() if seg.labels is not None else [0] * N
    scores = seg.scores.cpu().numpy().tolist() if seg.scores is not None else [1.0] * N
    boxes  = seg.boxes.cpu().numpy().tolist()  if seg.boxes  is not None else [[0,0,0,0]] * N

    if palette is None:
        if class_names is not None:
            palette = make_palette(max(len(class_names), 1), seed=42)
        else:
            # No class info; we’ll color by instance index
            palette = make_palette(256, seed=42)

    # Overlay each instance (front-to-back)
    for i in range(N):
        m = masks_bool[i]
        if not m.any():
            continue
        cls_id = int(labels[i]) if i < len(labels) else 0
        color = tuple(int(c) for c in palette[cls_id % len(palette)]) if palette is not None else color_for_id(cls_id, seed=seed)
        overlay = np.full_like(out, color, dtype=np.uint8)
        _alpha_blend(out, overlay, m, alpha)

        if draw_boxes and seg.boxes is not None:
            lab = ""
            if draw_labels:
                name = (class_names[cls_id] if class_names and 0 <= cls_id < len(class_names) else str(cls_id))
                if seg.scores is not None:
                    lab = f"{name} {score_fmt.format(scores[i])}"
                else:
                    lab = f"{name}"
            _draw_box_and_label(out, boxes[i], lab, color, thickness=thickness)

    return out


def colorize_semantic(
    sem_labels: np.ndarray,
    *,
    palette: Optional[np.ndarray] = None,  # [K,3] BGR
    num_classes: Optional[int] = None,
    seed: int = 42,
    unknown_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Convert a dense label map [H,W] to a color image (BGR).

    If `palette` is None, a palette of length `num_classes` (or max label + 1)
    is generated. Unknown/negative labels map to `unknown_color`.
    """
    if hasattr(sem_labels, "detach"):
        sem_np = sem_labels.detach().cpu().numpy()
    else:
        sem_np = np.asarray(sem_labels)
    assert sem_np.ndim == 2, "sem_labels must be [H,W]"

    H, W = sem_np.shape
    max_lab = int(np.max(sem_np)) if sem_np.size else 0
    K = num_classes or (max_lab + 1)
    if palette is None:
        palette = make_palette(max(K, 1), seed=seed)  # [K,3] BGR

    out = np.zeros((H, W, 3), dtype=np.uint8)
    valid = sem_np >= 0
    # Clip labels to palette range (wrap-around to keep deterministic)
    labs = np.where(valid, sem_np % len(palette), -1)

    # Vectorized coloring: for each class id, set pixels
    for cid in np.unique(labs):
        if cid < 0:
            continue
        out[labs == cid] = palette[int(cid)]

    # Unknown/invalid pixels
    out[~valid] = np.array(unknown_color, dtype=np.uint8)
    return out


def overlay_semantic(
    frame_bgr: np.ndarray,
    sem_labels: np.ndarray,
    *,
    alpha: float = 0.5,
    palette: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """Overlay a colorized semantic map onto the BGR frame."""
    assert frame_bgr.dtype == np.uint8 and frame_bgr.ndim == 3
    color = colorize_semantic(sem_labels, palette=palette, num_classes=num_classes, seed=seed)
    out = frame_bgr.copy()
    _alpha_blend(out, color, np.ones(sem_labels.shape, dtype=bool), alpha)
    return out


def overlay_panoptic(
    frame_bgr: np.ndarray,
    seg: SegResult,
    *,
    alpha: float = 0.5,
    draw_things_boxes: bool = True,
    draw_labels: bool = True,
    score_fmt: str = "{:.2f}",
    thickness: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Overlay a panoptic map (unique color per segment) and optionally draw boxes for 'things'.

    Uses seg.panoptic_map and seg.segments_info. If seg.boxes/labels/scores exist,
    draws boxes/labels for those as 'things' instances.
    """
    assert frame_bgr.ndim == 3 and frame_bgr.dtype == np.uint8
    if getattr(seg, "panoptic_map", None) is None or seg.panoptic_map.numel() == 0:
        return frame_bgr.copy()

    H, W = frame_bgr.shape[:2]
    pan = seg.panoptic_map.cpu().numpy().astype(np.int64)
    out = frame_bgr.copy()

    # Build a deterministic color for each segment id
    seg_ids = np.unique(pan)
    color_map: Dict[int, Tuple[int, int, int]] = {}
    for sid in seg_ids:
        color_map[int(sid)] = color_for_id(int(sid), seed=seed)

    # Paint per segment via vectorized indexing
    overlay = np.zeros_like(out)
    mask_any = np.zeros((H, W), dtype=bool)
    for sid in seg_ids:
        m = (pan == sid)
        overlay[m] = color_map[int(sid)]
        mask_any |= m

    _alpha_blend(out, overlay, mask_any, alpha)

    # Optionally draw boxes/labels for "things" instances (already filtered by backend)
    if draw_things_boxes and seg.boxes is not None and seg.labels is not None:
        class_names: Optional[List[str]] = None
        if isinstance(seg.config, dict) and isinstance(seg.config.get("classes"), (list, tuple)):
            class_names = list(seg.config["classes"])  # type: ignore

        boxes = seg.boxes.cpu().numpy().tolist()
        labels = seg.labels.cpu().numpy().tolist()
        scores = seg.scores.cpu().numpy().tolist() if seg.scores is not None else [1.0] * len(boxes)

        for i, xyxy in enumerate(boxes):
            # Color by the segment under the box center if possible
            cx = int((xyxy[0] + xyxy[2]) * 0.5)
            cy = int((xyxy[1] + xyxy[3]) * 0.5)
            sid = int(pan[min(max(cy, 0), H - 1), min(max(cx, 0), W - 1)])
            color = color_map.get(sid, color_for_id(i, seed=seed))

            lab = ""
            if draw_labels:
                cls_id = int(labels[i])
                name = (class_names[cls_id] if class_names and 0 <= cls_id < len(class_names) else str(cls_id))
                if seg.scores is not None:
                    lab = f"{name} {score_fmt.format(scores[i])}"
                else:
                    lab = f"{name}"
            _draw_box_and_label(out, xyxy, lab, color, thickness=thickness)

    return out