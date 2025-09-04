"""
models.py – unified wrappers for multiple Object Detection (OD) models.

Supports: YOLOv8, DETR, RT-DETR, YOLOS, Faster R-CNN, SSD.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Union

import torch
import numpy as np
from PIL import Image

from branes_platform.nn.base import BranesModel

__all__ = ["ODModel", "get_supported_od_models"]


# ------------------------------ Registry ------------------------------- #
SUPPORTED_MODELS = [
    "yolo",
    "detr",
    "rt-detr",
    "yolos",
    "fasterrcnn",
    "ssd300",
]

def get_supported_od_models() -> list[str]:
    return SUPPORTED_MODELS


# ---------------------------- Main Wrapper ----------------------------- #
class ODModel(BranesModel):
    def __init__(
        self,
        model_name: str = "yolo",
        device: Union[str, torch.device, None] = None,
        compile_model: bool | dict[str, Any] = False,
        **kwargs,
    ) -> None:
        super().__init__(device)
        self.model_name = model_name.lower()
        self.config = {}

        if self.model_name == "yolo":
            from ultralytics import YOLO
            weight = kwargs.get("weight", "yolov8n.pt")
            self.model = YOLO(weight).to(self.device)
            self.model.fuse()
            self._compile_target = getattr(self.model, "model", self.model)
            self.config.update(dict(architecture="YOLOv8", weight=str(weight)))

        elif self.model_name == "detr":
            from transformers import DetrImageProcessor, DetrForObjectDetection
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(self.device)
            self._compile_target = self.model
            self.config.update(dict(architecture="DETR"))

        elif self.model_name == "rt-detr":
            from ultralytics import YOLO
            weight='rtdetr-l.pt'
            self.model = YOLO(weight).to(self.device)
            self.model.fuse()
            self._compile_target = getattr(self.model, "model", self.model)
            self.config.update({
                "architecture": "RT-DETR (Ultralytics)",
                "weight": str(weight),
                "input_format": "BGR uint8 (numpy)",
                "output_format": "[x1,y1,x2,y2,conf,cls]",
            })

        elif self.model_name == "yolos":
            from transformers import YolosImageProcessor, YolosForObjectDetection
            self.processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
            self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(self.device)
            self._compile_target = self.model
            self.config.update(dict(architecture="YOLOS"))

        elif self.model_name == "fasterrcnn":
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            self.model = fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
            self.model.eval()
            self._compile_target = self.model
            self.config.update(dict(architecture="Faster R-CNN"))

        elif self.model_name == "ssd300":
            from torchvision.models.detection import ssd300_vgg16
            self.model = ssd300_vgg16(pretrained=True).to(self.device)
            self.model.eval()
            self._compile_target = self.model
            self.config.update(dict(architecture="SSD300"))

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.model.eval()
        if compile_model:
            self.compile(**(compile_model if isinstance(compile_model, dict) else {}))

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def predict(
        self,
        frame_bgr: np.ndarray,
        conf_thres: float = 0.3,
        classes: Sequence[int] | None = None,
    ) -> torch.Tensor:
        image_rgb = frame_bgr[:, :, ::-1]  # BGR to RGB

        if self.model_name in  ["yolo","rt-detr","rtdetr"]:
            results = self.model.predict(image_rgb, conf=conf_thres, classes=classes, device=str(self.device), verbose=False)
            dets = [[*b.xyxy[0].tolist(), float(b.conf), int(b.cls)] for r in results for b in getattr(r, "boxes", [])]
            return torch.tensor(dets, dtype=torch.float32, device=self.device) if dets else torch.empty((0, 6), device=self.device)

        elif self.model_name in ["detr", "yolos"]:
            inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            if hasattr(self.processor, "post_process_object_detection"):
                res = self.processor.post_process_object_detection(outputs, threshold=conf_thres, target_sizes=[frame_bgr.shape[:2]])[0]
            else:
                res = self.processor.post_process(outputs, target_sizes=[frame_bgr.shape[:2]], threshold=conf_thres)[0]
            boxes = torch.column_stack((res["boxes"], res["scores"], res["labels"].float())).to(self.device)
            return boxes

        elif self.model_name in ["fasterrcnn", "ssd300"]:
            tensor = torch.from_numpy(image_rgb.copy()).permute(2,0,1).contiguous().float() / 255.0
            inputs = tensor.unsqueeze(0).to(self.device)
            outputs = self.model(inputs)[0]
            boxes = torch.column_stack((outputs["boxes"], outputs["scores"], outputs["labels"].float())).to(self.device)
            if conf_thres:
                mask = boxes[:, 4] > conf_thres
                boxes = boxes[mask]
            return boxes

        return torch.empty((0, 6), device=self.device)

    # --------------------------------------------------------------------- #
    def compile(self, **kwargs):
        if not hasattr(torch, "compile"):
            warnings.warn("torch.compile unavailable – skipping")
            return self
        try:
            compiled = torch.compile(self._compile_target, **kwargs)
            if hasattr(self.model, "model"):
                self.model.model = compiled
            else:
                self.model = compiled
        except Exception as e:
            warnings.warn(f"torch.compile failed: {e}")
        return self