"""
models.py – unified wrappers for Object Detection (OD) and Re‑ID networks.

Author: You
Date: 2025‑06‑25
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Union
import torch
import numpy as np
from PIL import Image

__all__ = [
    "BranesModel",
    "ODModel",
    "ReIDModel",
]


# --------------------------------------------------------------------------- #
#                               Base interface                                #
# --------------------------------------------------------------------------- #


class BranesModel(ABC):
    """Small interface that all vision models inherit from."""

    def __init__(self, device: Union[str, torch.device, None] = None) -> None:
        self.device: torch.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config: Dict[str, Any] = {}

    # ---- life‑cycle -------------------------------------------------------- #

    @abstractmethod
    def predict(self, *args, **kwargs):  # noqa: D401
        """Run the network and return predictions."""

    def compile(self, *args, **kwargs):  # noqa: D401
        """Optional torch.compile / TensorRT hook (no‑op by default)."""
        return self


# --------------------------------------------------------------------------- #
#                              Object detector                                #
# --------------------------------------------------------------------------- #


class ODModel(BranesModel):
    """Wrapper around YOLO‑v8/9 or DETR detectors."""

    def __init__(
        self,
        model_name: str = "yolo",
        device: Union[str, torch.device, None] = None,
        compile_model: bool | dict[str, Any] = False,
            **kwargs,
    ) -> None:
        super().__init__(device)
        self.model_name = model_name.lower()

        if self.model_name.startswith("yolo"):
            from ultralytics import YOLO

            weight = kwargs.get("weight", "yolov8n.pt")
            self.model = YOLO(weight).to(self.device)
            self.model.fuse()
            self._compile_target = getattr(self.model, "model", self.model)
            self.config.update(
                {
                    "architecture": "YOLO",
                    "weight": str(weight),
                    "input_format": "BGR uint8 (numpy)",
                    "output_format": "[x1,y1,x2,y2,conf,cls]",
                }
            )

        elif self.model_name == "detr":
            from transformers import DetrImageProcessor, DetrForObjectDetection

            self.processor = DetrImageProcessor.from_pretrained(
                "facebook/detr-resnet-50"
            )
            self.model = (
                DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
                .to(self.device)
                .eval()
            )
            self._compile_target = self.model
            self.config.update(
                {
                    "architecture": "DETR-ResNet50",
                    "weight": "facebook/detr-resnet-50",
                    "input_format": "BGR uint8 (numpy)",
                    "output_format": "[x1,y1,x2,y2,conf,cls]",
                }
            )
        else:
            raise ValueError(f"Unsupported OD model: {model_name}")

        self.model.eval()
        if compile_model:
            self.compile(**(compile_model if isinstance(compile_model, dict) else {}))

    # --------------------------------------------------------------------- #
    #                              public API                               #
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def predict(
            self,
            frame_bgr: np.ndarray,
            conf_thres: float = 0.3,
            classes: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return (N,6) tensor [x1,y1,x2,y2,conf,cls] on self.device."""
        if self.model_name.startswith("yolo"):
            results = self.model.predict(
                frame_bgr,
                conf=conf_thres,
                classes=classes,
                device=str(self.device),
                half=self.device.type == "cuda",
                verbose=False,
            )
            dets: list[list[float]] = []
            for r in results:
                for b in getattr(r, "boxes", []):
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    dets.append([x1, y1, x2, y2, float(b.conf), int(b.cls)])
            if dets:
                return torch.tensor(dets, dtype=torch.float32, device=self.device)
            # print(type(self.od.model.model))
            return torch.empty((0, 6), dtype=torch.float32, device=self.device)

        # --- DETR branch ---------------------------------------------------- #
        from PIL import Image

        inputs = self.processor(
            images=frame_bgr[:, :, ::-1], return_tensors="pt"
        ).to(self.device)
        out = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            out, threshold=conf_thres, target_sizes=[frame_bgr.shape[:2]]
        )[0]
        boxes = torch.column_stack(
            (
                res["boxes"].to(self.device),
                res["scores"].to(self.device),
                res["labels"].float().to(self.device),
            )
        )
        if classes is not None:
            mask = torch.isin(boxes[:, 5].int(), torch.as_tensor(classes))
            boxes = boxes[mask]
        # print(type(self.model))  # Should show:
        return boxes

        # ------------------------------------------------------------------ #

    def compile(self, **kwargs):  # type: ignore[override]
        """JIT‑compile the underlying *torch* module with `torch.compile()`.
        We *skip* this step on PyTorch < 2.0.
        """
        if not hasattr(torch, "compile"):
            warnings.warn("torch.compile unavailable – skipping compilation")
            return self
        try:
            compiled = torch.compile(self._compile_target, **kwargs)
            if self.model_name.startswith("yolo") and hasattr(self.model, "model"):
                # Hot‑swap the internals YOLO expects
                self.model.model = compiled
            else:
                self.model = compiled  # DETR path
            print(f"Compiling {self.model_name} model with torch.compile()")

        except Exception as e:  # pragma: no‑cover – backend quirks
            warnings.warn(f"torch.compile failed for {self.model_name}: {e}")
        return self


# --------------------------------------------------------------------------- #
#                                 ReID                                       #
# --------------------------------------------------------------------------- #


class ReIDModel(BranesModel):
    """Image Re‑Identification encoder (currently CLIP ViT‑B/32)."""

    def __init__(
            self, model_name: str = "clip", device: Union[str, torch.device, None] = None,
            compile_model: bool | dict[str, Any] = False
    ) -> None:
        super().__init__(device)
        self.model_name = model_name.lower()

        if self.model_name == "clip":
            import open_clip

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self.model = self.model.visual.eval().to(self.device)
            self._compile_target = self.model
            self.config.update(
                {
                    "architecture": "CLIP-ViT-B/32",
                    "pretrained": "laion2b_s34b_b79k",
                    "embed_dim": 512,
                    "input_size": 224,
                }
            )
        else:
            raise ValueError(f"Unsupported ReID model: {model_name}")

        if compile_model:
            self.compile(**(compile_model if isinstance(compile_model, dict) else {}))

    def compile(self, **kwargs):  # type: ignore[override]
        if not hasattr(torch, "compile"):
            warnings.warn("torch.compile unavailable – skipping compilation")
            return self
        try:
            print(f"Compiling ReID model {self.model_name} with torch.compile()")
            self.model = torch.compile(self._compile_target, **kwargs)
        except Exception as e:
            warnings.warn(f"torch.compile failed for ReID model: {e}")
        return self

    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def predict(
            self,
            frame_bgr: np.ndarray,
            boxes_xyxy: Union[torch.Tensor, np.ndarray, List[Sequence[float]]],
    ) -> torch.Tensor:
        """Return L2‑normalised (N,D) features on self.device."""
        D = self.config.get("embed_dim", 512)
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return torch.empty((0, D), dtype=torch.float32, device=self.device)
        boxes = torch.as_tensor(boxes_xyxy, dtype=torch.int64)

        H, W, _ = frame_bgr.shape
        crops: List[Image.Image] = []
        for x1, y1, x2, y2 in boxes.tolist():
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            x2, y2 = min(int(x2), W - 1), min(int(y2), H - 1)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2, ::-1]
            crops.append(Image.fromarray(crop))
        if not crops:
            return torch.empty((0, D), dtype=torch.float32, device=self.device)
        batch = torch.stack([self.preprocess(img) for img in crops]).to(self.device)
        feats = self.model(batch)
        # print(type(self.model))
        return torch.nn.functional.normalize(feats, dim=1)
