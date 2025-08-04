"""
models.py – unified wrappers for segmentation (Instance/Semantic/Panoptic).

Author : You
Created: 2025-07-31

Usage example
-------------
>>> from branes_platform.nn.seg.models import SegModel
>>> seg = SegModel(task="instance", backend="yolo", device="cpu")
>>> res = seg.predict(frame_bgr, conf_thres=0.4, mask_format="rle")
>>> print(res.boxes.shape, len(res.rles), res.labels.shape)
>>> od_like = res.to_od()  # (N,6) [x1,y1,x2,y2,conf,cls]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import warnings
import numpy as np
import torch
import torch.nn.functional as F

try:
    # Optional, used for compact/standard COCO RLE
    from pycocotools import mask as pycoco_mask  # type: ignore
    _HAS_COCO = True
except Exception:
    _HAS_COCO = False

# Keep parity with your existing base class
from branes_platform.nn.base import BranesModel

__all__ = [
    "SegModel",
    "SegResult",
]


# --------------------------------------------------------------------------- #
#                               Result object                                 #
# --------------------------------------------------------------------------- #

@dataclass
class SegResult:
    """Standardised output for segmentation across tasks/backends.

    Notes
    -----
    Exactly one of (masks | rles | polys) is populated for instance/panoptic
    masks, depending on `mask_format` passed to `SegModel.predict()`.
    """
    task: Literal["instance", "semantic", "panoptic"]

    # Instance / panoptic "things" as a list (optional for semantic)
    boxes: Optional[torch.FloatTensor] = None     # [N,4] xyxy
    scores: Optional[torch.FloatTensor] = None    # [N]
    labels: Optional[torch.LongTensor] = None     # [N]

    # Per-instance masks (one of these will be set for instance/panoptic)
    masks: Optional[torch.BoolTensor] = None      # [N,H,W] if mask_format="bitmap"
    rles: Optional[List[Dict[str, Any]]] = None   # COCO-style RLEs if mask_format="rle"
    polys: Optional[List[List[np.ndarray]]] = None  # polygons per instance

    # Semantic output (dense)
    sem_labels: Optional[torch.LongTensor] = None  # [H,W]

    # Panoptic consolidated map + metadata
    panoptic_map: Optional[torch.IntTensor] = None  # [H,W], integer segment ids
    segments_info: Optional[List[Dict[str, Any]]] = None  # [{"id", "category_id", "isthing", "score"}, ...]

    # Metadata / extras
    config: Dict[str, Any] = field(default_factory=dict)

    # ----------------------------- helpers -------------------------------- #

    def to_od(self) -> torch.FloatTensor:
        """Return (N,6) float32 [x1,y1,x2,y2,conf,cls] if available."""
        if self.boxes is None or self.scores is None or self.labels is None:
            return torch.empty((0, 6), dtype=torch.float32)
        boxes = self.boxes.to(torch.float32)
        scores = self.scores.to(torch.float32).unsqueeze(1)
        labels = self.labels.to(torch.float32).unsqueeze(1)
        return torch.cat([boxes, scores, labels], dim=1)

    def num_instances(self) -> int:
        if self.boxes is not None:
            return int(self.boxes.shape[0])
        if self.rles is not None:
            return len(self.rles)
        if self.masks is not None:
            return int(self.masks.shape[0])
        if self.polys is not None:
            return len(self.polys)
        return 0


# --------------------------------------------------------------------------- #
#                               Utilities                                     #
# --------------------------------------------------------------------------- #

def _ensure_hwc_bgr_uint8(frame_bgr: np.ndarray) -> Tuple[int, int]:
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be HxWx3 uint8 ndarray (BGR).")
    if frame_bgr.dtype != np.uint8:
        raise ValueError(f"frame_bgr must be uint8, got {frame_bgr.dtype}.")
    H, W = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    return H, W


def _resize_masks_to(m: torch.Tensor, out_h: int, out_w: int, thr: float = 0.5) -> torch.BoolTensor:
    """Resize mask tensor [N,h,w] -> [N,out_h,out_w] using bilinear then threshold."""
    if m.ndim != 3:
        raise ValueError("masks must be [N,h,w]")
    if (m.shape[-2] == out_h) and (m.shape[-1] == out_w):
        # ensure boolean
        return (m > 0.5) if m.dtype != torch.bool else m
    # Interpolate expects NCHW
    m_float = m.float().unsqueeze(1)  # [N,1,h,w]
    m_up = F.interpolate(m_float, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return (m_up.squeeze(1) >= thr)


def _bitmap_to_rle_single(mask_np: np.ndarray) -> Dict[str, Any]:
    """Convert a single HxW bool/0-1 mask to COCO-style RLE dict."""
    if _HAS_COCO:
        m_fortran = np.asfortranarray(mask_np.astype(np.uint8))
        rle = pycoco_mask.encode(m_fortran)
        # pycocotools returns counts as bytes; ensure json-serializable
        rle["counts"] = rle["counts"].decode("ascii")  # type: ignore
        return {"size": rle["size"], "counts": rle["counts"]}
    # Fallback simple RLE (uncompressed; still COCO-like dict)
    # Flatten in column-major order (COCO convention)
    m = mask_np.astype(np.uint8).T.flatten()
    counts: List[int] = []
    prev = 0
    run_len = 0
    for pix in m:
        if pix == prev:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
            prev = pix
    counts.append(run_len)
    return {"size": [int(mask_np.shape[0]), int(mask_np.shape[1])], "counts": counts}


def bitmaps_to_rles(masks: torch.BoolTensor) -> List[Dict[str, Any]]:
    """Convert [N,H,W] bool tensor to list of RLE dicts."""
    if masks.numel() == 0:
        return []
    masks_np = masks.cpu().numpy().astype(bool)
    return [_bitmap_to_rle_single(m) for m in masks_np]


def rles_to_bitmaps(rles: List[Dict[str, Any]], out_h: int, out_w: int) -> torch.BoolTensor:
    """Decode a list of COCO RLE dicts to [N,H,W] bool tensor."""
    if not rles:
        return torch.zeros((0, out_h, out_w), dtype=torch.bool)
    if _HAS_COCO:
        # pycocotools expects list of RLEs; ensure bytes for counts
        rles_enc = []
        for r in rles:
            rr = {"size": r["size"], "counts": r["counts"]}
            if isinstance(rr["counts"], str):
                rr["counts"] = rr["counts"].encode("ascii")
            rles_enc.append(rr)
        decoded = pycoco_mask.decode(rles_enc)  # HxWxN or HxW for N=1
        if decoded.ndim == 2:
            decoded = decoded[..., None]
        decoded = np.moveaxis(decoded, -1, 0)  # N,H,W
        return torch.from_numpy(decoded.astype(bool))
    # Fallback decoder for simple RLE counts
    out = np.zeros((len(rles), out_h, out_w), dtype=np.uint8)
    for i, r in enumerate(rles):
        counts = r["counts"]
        if isinstance(counts, str):
            raise RuntimeError("Compressed RLE decoding requires pycocotools.")
        size = r["size"]
        H, W = int(size[0]), int(size[1])
        if (H, W) != (out_h, out_w):
            H, W = out_h, out_w  # trust the target dims
        arr = np.zeros(H * W, dtype=np.uint8)
        val = 0
        idx = 0
        for run in counts:
            if run > 0:
                arr[idx : idx + run] = val
                idx += run
                val = 1 - val
        arr = arr.reshape((W, H)).T  # inverse of column-major flatten
        out[i] = arr
    return torch.from_numpy(out.astype(bool))


def masks_to_polygons(masks: torch.BoolTensor) -> List[List[np.ndarray]]:
    """Convert [N,H,W] boolean masks to polygons (list of contours per instance).

    Uses OpenCV if available; otherwise returns empty lists.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return [[] for _ in range(int(masks.shape[0]))]

    polys: List[List[np.ndarray]] = []
    masks_np = masks.cpu().numpy().astype(np.uint8)
    for m in masks_np:
        contours, _ = cv2.findContours(m, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        polys.append(contours)
    return polys


# --------------------------------------------------------------------------- #
#                                Backends                                     #
# --------------------------------------------------------------------------- #

class _InstanceSeg_YOLO:
    """YOLOv8/9 instance segmentation backend (Ultralytics)."""

    def __init__(self, device: Union[str, torch.device], compile_model: bool, **kwargs) -> None:
        from ultralytics import YOLO

        weight = kwargs.get("weight", "yolov8n-seg.pt")
        self.model = YOLO(weight).to(device)
        # Fuse layers if available (safe on CPU too)
        if hasattr(self.model, "fuse"):
            try:
                self.model.fuse()
            except Exception:
                pass

        # Try torch.compile on internal torch module if requested and available
        self._compile_target = getattr(self.model, "model", self.model)
        if compile_model and hasattr(torch, "compile"):
            try:
                compiled = torch.compile(self._compile_target)
                if hasattr(self.model, "model"):
                    self.model.model = compiled
                else:
                    self.model = compiled
                print("Compiling YOLO-seg model with torch.compile()")
            except Exception as e:
                warnings.warn(f"torch.compile failed for YOLO-seg: {e}")

        # Config metadata (class names from model)
        names = getattr(self.model, "names", None)
        if names is None and hasattr(self.model, "model") and hasattr(self.model.model, "names"):
            names = self.model.model.names
        if isinstance(names, dict):
            # Ultralytics sometimes stores {id: "name"}
            names = [names[k] for k in sorted(names.keys())]
        self.config: Dict[str, Any] = {
            "architecture": "YOLOv8/9-seg",
            "weight": str(weight),
            "input_format": "BGR uint8 (numpy)",
            "mask_default": "rle",
            "classes": names,
        }

        self.device = torch.device(device)

    @torch.no_grad()
    def predict(
        self,
        frame_bgr: np.ndarray,
        conf_thres: float = 0.3,
        classes: Optional[Sequence[int]] = None,
        mask_format: Literal["bitmap", "rle", "polygons"] = "rle",
        mask_thr: float = 0.5,
    ) -> SegResult:
        H, W = _ensure_hwc_bgr_uint8(frame_bgr)

        # Run the model (Ultralytics accepts np.ndarray directly)
        results = self.model.predict(
            frame_bgr,
            conf=conf_thres,
            classes=classes,
            device=str(self.device),
            half=False,           # CPU-friendly; set True automatically on CUDA if you later want
            verbose=False,
        )

        # Consolidate all detections in the (single) image
        boxes_list: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []   # each: [n_i, h, w]
        polys_list: List[List[np.ndarray]] = []

        total = 0
        for r in results:
            # Boxes
            b = getattr(r, "boxes", None)
            if b is None or b.shape[0] == 0:
                continue
            xyxy = b.xyxy  # (n,4) on CPU (ultralytics returns torch)
            conf = b.conf  # (n,)
            cls = b.cls    # (n,)
            boxes_list.append(xyxy.to(torch.float32))
            scores_list.append(conf.to(torch.float32))
            labels_list.append(cls.to(torch.long))

            # Masks
            m_obj = getattr(r, "masks", None)
            # Prefer Ultralytics' own polygons if requested
            if mask_format == "polygons" and m_obj is not None and hasattr(m_obj, "xy"):
                polys = m_obj.xy  # list of [K,2] arrays per instance, likely already in original scale
                polys_list.extend(polys if isinstance(polys, list) else [])
            elif m_obj is not None and hasattr(m_obj, "data") and m_obj.data is not None:
                m = m_obj.data  # [n, mh, mw], float/bool tensor
                # Align to original frame size if needed
                m = m.cpu()
                m = _resize_masks_to(m, H, W, thr=mask_thr)  # -> bool [n, H, W]
                masks_list.append(m)
            else:
                # No masks available—for safety, append empty to keep indexing consistent
                masks_list.append(torch.zeros((xyxy.shape[0], H, W), dtype=torch.bool))

            total += int(xyxy.shape[0])

        if total == 0:
            return SegResult(
                task="instance",
                boxes=torch.empty((0, 4), dtype=torch.float32),
                scores=torch.empty((0,), dtype=torch.float32),
                labels=torch.empty((0,), dtype=torch.long),
                masks=torch.empty((0, H, W), dtype=torch.bool) if mask_format == "bitmap" else None,
                rles=[] if mask_format == "rle" else None,
                polys=[] if mask_format == "polygons" else None,
                config=self.config.copy(),
            )

        # Concatenate across any internal splits (usually 1 image => 1 result)
        boxes = torch.cat(boxes_list, dim=0) if boxes_list else torch.empty((0, 4), dtype=torch.float32)
        scores = torch.cat(scores_list, dim=0) if scores_list else torch.empty((0,), dtype=torch.float32)
        labels = torch.cat(labels_list, dim=0) if labels_list else torch.empty((0,), dtype=torch.long)

        # Gather masks/polys according to requested format
        out_masks: Optional[torch.BoolTensor] = None
        out_rles: Optional[List[Dict[str, Any]]] = None
        out_polys: Optional[List[List[np.ndarray]]] = None

        if mask_format == "polygons":
            # If polygons came from Ultralytics, they are already in image scale.
            # If not available, fall back to bitmap->contours.
            if polys_list and len(polys_list) == int(boxes.shape[0]):
                out_polys = polys_list
            else:
                # Fall back to bitmap path
                if masks_list:
                    masks = torch.cat(masks_list, dim=0)
                    out_polys = masks_to_polygons(masks)
                else:
                    out_polys = []
        elif mask_format == "bitmap":
            if masks_list:
                out_masks = torch.cat(masks_list, dim=0)
            else:
                out_masks = torch.zeros((int(boxes.shape[0]), H, W), dtype=torch.bool)
        else:  # "rle" (default)
            if masks_list:
                m = torch.cat(masks_list, dim=0)  # [N,H,W]
                out_rles = bitmaps_to_rles(m)
            else:
                out_rles = []

        return SegResult(
            task="instance",
            boxes=boxes,
            scores=scores,
            labels=labels,
            masks=out_masks,
            rles=out_rles,
            polys=out_polys,
            config=self.config.copy(),
        )


class _SemanticSeg_SegFormer:
    """SegFormer backend (semantic segmentation) – scaffold for next milestone."""
    def __init__(self, device: Union[str, torch.device], compile_model: bool, **kwargs) -> None:
        self.device = torch.device(device)
        self.config: Dict[str, Any] = {
            "architecture": "SegFormer",
            "weight": kwargs.get("weight", "nvidia/segformer-b0-finetuned-ade-512-512"),
            "input_format": "BGR uint8 (numpy)",
            "classes": None,  # to be populated when implemented
        }
        raise NotImplementedError("Semantic backend not implemented yet (Milestone B).")

    @torch.no_grad()
    def predict(self, *args, **kwargs) -> SegResult:
        raise NotImplementedError


class _PanopticSeg_Mask2Former:
    """Mask2Former backend (panoptic segmentation) – scaffold for later milestone."""
    def __init__(self, device: Union[str, torch.device], compile_model: bool, **kwargs) -> None:
        self.device = torch.device(device)
        self.config: Dict[str, Any] = {
            "architecture": "Mask2Former",
            "weight": kwargs.get("weight", "facebook/mask2former-swin-base-ade20k-panoptic"),
            "input_format": "BGR uint8 (numpy)",
            "classes": None,  # to be populated when implemented
        }
        raise NotImplementedError("Panoptic backend not implemented yet (Milestone C).")

    @torch.no_grad()
    def predict(self, *args, **kwargs) -> SegResult:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#                                Public API                                   #
# --------------------------------------------------------------------------- #

class SegModel(BranesModel):
    """Unified segmentation interface.

    Parameters
    ----------
    task : {"instance","semantic","panoptic"}
        Segmentation task to run.
    backend : str
        For now: "yolo" (instance). "segformer" and "mask2former" are scaffolded.
    device : str | torch.device | None
        Device string; defaults to BranesModel heuristic.
    compile_model : bool or dict
        Use torch.compile on supported backends where available.
    **kwargs
        Backend-specific options. Common:
        - weight: model checkpoint path or hub id
        - mask_thr: float for binarising probability masks
        - input_size, etc. (future)

    Notes
    -----
    - Input format: BGR uint8 (numpy), HxWx3 – same as ODModel.
    - For CPU-first runs we disable half precision.
    """

    def __init__(
        self,
        task: Literal["instance", "semantic", "panoptic"] = "instance",
        backend: str = "yolo",
        device: Union[str, torch.device, None] = None,
        compile_model: bool | Dict[str, Any] = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(device)
        self.task = task.lower()
        self.backend = backend.lower()

        # Normalize compile flag
        compile_flag: bool
        compile_kwargs: Dict[str, Any]
        if isinstance(compile_model, dict):
            compile_flag, compile_kwargs = True, compile_model
        else:
            compile_flag, compile_kwargs = bool(compile_model), {}

        # Select backend
        if self.task == "instance" and self.backend in ("yolo", "yolo-seg"):
            self.impl = _InstanceSeg_YOLO(device=self.device, compile_model=compile_flag, **kwargs)
        elif self.task == "semantic" and self.backend in ("segformer",):
            self.impl = _SemanticSeg_SegFormer(device=self.device, compile_model=compile_flag, **kwargs)
        elif self.task == "panoptic" and self.backend in ("mask2former",):
            self.impl = _PanopticSeg_Mask2Former(device=self.device, compile_model=compile_flag, **kwargs)
        else:
            raise ValueError(f"Unsupported combination task={self.task} backend={self.backend}")

        # Merge config from backend
        self.config.update(getattr(self.impl, "config", {}))
        # Record device
        self.model = getattr(self.impl, "model", None)

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict(
        self,
        frame_bgr: np.ndarray,
        conf_thres: float = 0.3,
        classes: Optional[Sequence[int]] = None,
        mask_format: Literal["bitmap", "rle", "polygons"] = "rle",
        mask_thr: float = 0.5,
    ) -> SegResult:
        """Run segmentation on a single frame.

        Returns
        -------
        SegResult
            Standardised container with per-task outputs.
        """
        return self.impl.predict(
            frame_bgr=frame_bgr,
            conf_thres=conf_thres,
            classes=classes,
            mask_format=mask_format,
            mask_thr=mask_thr,
        )


class _SemanticSeg_SegFormer:
    """SegFormer backend (semantic segmentation) – ADE20K by default."""

    def __init__(self, device: Union[str, torch.device], compile_model: bool, **kwargs) -> None:
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        weight = kwargs.get("weight", "nvidia/segformer-b0-finetuned-ade-512-512")
        self.device = torch.device(device)

        # Preprocessor + model
        self.processor = AutoImageProcessor.from_pretrained(weight)
        self.model = SegformerForSemanticSegmentation.from_pretrained(weight).to(self.device).eval()

        # Optional compile (often no benefit on CPU, but keep parity with your API)
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                print("Compiling SegFormer model with torch.compile()")
            except Exception as e:
                warnings.warn(f"torch.compile failed for SegFormer: {e}")

        # Class names from config.id2label (ADE20K = 150)
        cfg = self.model.config
        names = None
        try:
            if getattr(cfg, "id2label", None):
                # Prefer numeric order 0..num_labels-1 if available
                if hasattr(cfg, "num_labels") and isinstance(cfg.num_labels, int):
                    names = [cfg.id2label[i] for i in range(cfg.num_labels)]
                else:
                    # Fallback: sort keys (may be strings) numerically if possible
                    keys = list(cfg.id2label.keys())
                    try:
                        keys_sorted = sorted(keys, key=lambda k: int(k))
                    except Exception:
                        keys_sorted = sorted(keys)
                    names = [cfg.id2label[k] for k in keys_sorted]
        except Exception:
            names = None

        self.config: Dict[str, Any] = {
            "architecture": "SegFormer",
            "variant": getattr(cfg, "name_or_path", None) or "segformer-b0",
            "weight": str(weight),
            "dataset": "ADE20K",
            "num_labels": getattr(cfg, "num_labels", None),
            "input_format": "BGR uint8 (numpy)",
            "classes": names,  # list[str] | None
        }

    @torch.no_grad()
    def predict(
        self,
        frame_bgr: np.ndarray,
        conf_thres: float = 0.3,  # unused for semantic
        classes: Optional[Sequence[int]] = None,  # not applied; see note below
        mask_format: Literal["bitmap", "rle", "polygons"] = "rle",  # unused for semantic
        mask_thr: float = 0.5,  # unused for semantic
    ) -> SegResult:
        """Return a dense label map [H,W] with class indices (torch.long)."""
        from PIL import Image

        H, W = _ensure_hwc_bgr_uint8(frame_bgr)
        rgb = frame_bgr[:, :, ::-1]
        pil = Image.fromarray(rgb)

        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # Resize/align to the original HxW
        sem = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[(H, W)])[0]
        # Ensure torch.long on CPU
        sem_t = (torch.from_numpy(sem) if isinstance(sem, np.ndarray) else sem.to("cpu")).to(torch.long)

        if classes is not None:
            # We return the full label map (dense semantics). If you want to mask to a subset,
            # we could add a flag later (e.g., set other classes to 0 or -1).
            warnings.warn("`classes` is ignored for semantic segmentation; returning full label map.")

        return SegResult(
            task="semantic",
            sem_labels=sem_t,           # [H,W], torch.long
            config=self.config.copy(),  # includes class names under "classes"
        )


class _PanopticSeg_Mask2Former:
    """Mask2Former backend (panoptic segmentation, COCO Panoptic by default)."""

    def __init__(self, device: Union[str, torch.device], compile_model: bool, **kwargs) -> None:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        weight = kwargs.get("weight", "facebook/mask2former-swin-base-coco-panoptic")
        self.device = torch.device(device)

        # Preprocessor + model
        self.processor = AutoImageProcessor.from_pretrained(weight)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(weight).to(self.device).eval()

        # Optional compile (often limited benefit on CPU; keep parity with your API)
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                print("Compiling Mask2Former model with torch.compile()")
            except Exception as e:
                warnings.warn(f"torch.compile failed for Mask2Former: {e}")

        # Class names (COCO Panoptic = 133)
        cfg = self.model.config
        names = None
        try:
            if getattr(cfg, "id2label", None):
                if hasattr(cfg, "num_labels") and isinstance(cfg.num_labels, int):
                    names = [cfg.id2label[i] for i in range(cfg.num_labels)]
                else:
                    keys = list(cfg.id2label.keys())
                    try:
                        keys_sorted = sorted(keys, key=lambda k: int(k))
                    except Exception:
                        keys_sorted = sorted(keys)
                    names = [cfg.id2label[k] for k in keys_sorted]
        except Exception:
            names = None

        self.config: Dict[str, Any] = {
            "architecture": "Mask2Former",
            "variant": getattr(cfg, "name_or_path", None) or "swin-base",
            "weight": str(weight),
            "dataset": "COCO Panoptic",
            "num_labels": getattr(cfg, "num_labels", None),
            "input_format": "BGR uint8 (numpy)",
            "classes": names,  # list[str] | None
        }

    @torch.no_grad()
    def predict(
        self,
        frame_bgr: np.ndarray,
        conf_thres: float = 0.3,  # Not used for panoptic post-process
        classes: Optional[Sequence[int]] = None,  # Applied to "things" instance list only
        mask_format: Literal["bitmap", "rle", "polygons"] = "rle",
        mask_thr: float = 0.5,  # Not used (panoptic masks are discrete)
    ) -> SegResult:
        """Return panoptic map + segments_info + 'things' instance list."""
        from PIL import Image

        H, W = _ensure_hwc_bgr_uint8(frame_bgr)
        rgb = frame_bgr[:, :, ::-1]
        pil = Image.fromarray(rgb)

        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        post = self.processor.post_process_panoptic_segmentation(outputs, target_sizes=[(H, W)])[0]

        # panoptic map (segment ids per pixel)
        seg = post.get("segmentation")
        if isinstance(seg, torch.Tensor):
            pan_map = seg.to("cpu").to(torch.int32)
        else:
            pan_map = torch.from_numpy(np.array(seg, dtype=np.int32))

        # segments_info: normalize keys and keep as JSON-friendly dicts
        segs_info_raw = post.get("segments_info", []) or []
        segments_info: List[Dict[str, Any]] = []
        for s in segs_info_raw:
            cat_id = s.get("category_id", s.get("label_id"))
            segments_info.append({
                "id": int(s.get("id")),
                "category_id": int(cat_id) if cat_id is not None else None,
                "isthing": bool(s.get("isthing")) if "isthing" in s else None,
                "score": float(s.get("score", 1.0)),
            })

        # Build a "things" instance list (boxes/masks/scores/labels)
        # If 'isthing' is provided, we include only isthing=True. Otherwise, include all segments.
        def _mask_to_xyxy(m: torch.BoolTensor) -> torch.FloatTensor:
            ys, xs = torch.where(m)
            if ys.numel() == 0:
                return torch.zeros(4, dtype=torch.float32)
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

        boxes_list: List[torch.FloatTensor] = []
        scores_list: List[float] = []
        labels_list: List[int] = []
        bitmaps_for_output: List[torch.BoolTensor] = []  # used if mask_format in {"bitmap","polygons"}
        rles_for_output: List[Dict[str, Any]] = []       # used if mask_format == "rle"

        for s in segments_info:
            # Filter to "things" if flag exists
            if s["isthing"] is not None and not s["isthing"]:
                continue
            cat_id = s["category_id"]
            if classes is not None and cat_id not in classes:
                continue

            seg_id = s["id"]
            mask_bool = (pan_map == int(seg_id))
            # Skip empty masks (shouldn't happen, but be safe)
            if not bool(mask_bool.any()):
                continue

            # Box from mask
            boxes_list.append(_mask_to_xyxy(mask_bool))
            scores_list.append(s["score"])
            labels_list.append(int(cat_id) if cat_id is not None else -1)

            # Prepare masks in requested format
            if mask_format == "bitmap" or mask_format == "polygons":
                bitmaps_for_output.append(mask_bool)
            else:  # "rle"
                mask_np = mask_bool.numpy()
                rles_for_output.append(_bitmap_to_rle_single(mask_np))

        # Consolidate instance tensors
        if boxes_list:
            boxes = torch.stack(boxes_list, dim=0)
            scores = torch.tensor(scores_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.long)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            scores = torch.empty((0,), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.long)

        out_masks: Optional[torch.BoolTensor] = None
        out_rles: Optional[List[Dict[str, Any]]] = None
        out_polys: Optional[List[List[np.ndarray]]] = None

        if mask_format == "bitmap":
            out_masks = torch.stack(bitmaps_for_output, dim=0) if bitmaps_for_output else torch.zeros((0, H, W), dtype=torch.bool)
        elif mask_format == "polygons":
            if bitmaps_for_output:
                tmp = torch.stack(bitmaps_for_output, dim=0)
                out_polys = masks_to_polygons(tmp)
            else:
                out_polys = []
        else:  # "rle"
            out_rles = rles_for_output

        return SegResult(
            task="panoptic",
            boxes=boxes if boxes.numel() else torch.empty((0, 4), dtype=torch.float32),
            scores=scores if scores.numel() else torch.empty((0,), dtype=torch.float32),
            labels=labels if labels.numel() else torch.empty((0,), dtype=torch.long),
            masks=out_masks,
            rles=out_rles,
            polys=out_polys,
            panoptic_map=pan_map.to(torch.int32),
            segments_info=segments_info,
            config=self.config.copy(),
        )