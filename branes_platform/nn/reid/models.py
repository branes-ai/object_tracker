"""
reid_models.py – unified wrappers for multiple ReID models.

Supports: CLIP (ViT-B/32, ViT-L/14), DINOv2, ViT, ResNet50-ReID.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Sequence, Union

import cv2
import torch
import numpy as np
from PIL import Image

__all__ = [
    "ReIDModel",
    "get_supported_reid_models"
]

from branes_platform.nn.base import BranesModel

SUPPORTED_MODELS = [
    "clip_vit_b32",
    "clip_vit_l14",
    "dinov2_vits14",
    "vit_b_16",
    "resnet50_reid",
]

def get_supported_reid_models() -> list[str]:
    return SUPPORTED_MODELS


class ReIDModel(BranesModel):
    def __init__(
        self,
        model_name: str = "clip_vit_b32",
        device: Union[str, torch.device, None] = None,
        compile_model: bool | dict[str, Any] = False,
    ) -> None:
        super().__init__(device)
        self.model_name = model_name.lower()
        self.config = {}

        if self.model_name.startswith("clip"):
            import open_clip
            arch = "ViT-B-32" if "b32" in self.model_name else "ViT-L-14"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                arch, pretrained="openai"
            )
            self.model = self.model.visual.eval().to(self.device)
            self._compile_target = self.model
            self.config.update(dict(architecture=f"CLIP-{arch}", embed_dim=self.model.output_dim, input_size=224))

        elif self.model_name.startswith("dinov2"):
            from transformers import AutoImageProcessor, AutoModel
            ckpt = "facebook/dinov2-small"  # s14
            self.processor = AutoImageProcessor.from_pretrained(ckpt)
            self.model = AutoModel.from_pretrained(ckpt).to(self.device).eval()
            self._compile_target = self.model
            self.config.update({"architecture": "DINOv2-small", "embed_dim": 384, "input_size": 224})

        elif self.model_name == "vit_b_16":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            m = vit_b_16(weights=weights)
            m.heads = torch.nn.Identity()  # <-- get penultimate features (768)
            self.preprocess = weights.transforms()
            self.model = m.to(self.device).eval()
            self._compile_target = self.model
            self.config.update({"architecture": "ViT-B/16", "embed_dim": 768, "input_size": 224})

        elif self.model_name in {"resnet50_reid", "resnet50"}:
            import timm
            self.model = timm.create_model("resnet50", pretrained=True, num_classes=0)  # global pooled 2048-D
            self.model.eval().to(self.device)
            from torchvision import transforms as T
            self.preprocess = T.Compose([
                T.Resize((256, 128)),  # typical ReID aspect
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._compile_target = self.model
            self.config.update(
                {"architecture": "ResNet50 (timm, penultimate)", "embed_dim": 2048, "input_size": (256, 128)})

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
            warnings.warn(f"torch.compile failed for ReID model {self.model_name}: {e}")
        return self

    @torch.no_grad()
    def predict(
        self,
        frame_bgr: np.ndarray,
        boxes_xyxy: Union[torch.Tensor, np.ndarray, List[Sequence[float]]],
    ) -> torch.Tensor:
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
            crop_bgr = frame_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)  # contiguous, no negative stride
            crops.append(Image.fromarray(crop_rgb))

        if not crops:
            return torch.empty((0, D), dtype=torch.float32, device=self.device)

        if self.model_name.startswith("dinov2"):
            # processor handles resize/normalize; returns pixel_values (N, C, H, W)
            inputs = self.processor(images=crops, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            outputs = self.model(pixel_values=pixel_values)
            # CLS token embedding
            feats = outputs.last_hidden_state[:, 0]  # (N, D=384 for small)
            return torch.nn.functional.normalize(feats, dim=1)

        batch = torch.stack([self.preprocess(img) for img in crops]).to(self.device)
        feats = self.model(batch)
        return torch.nn.functional.normalize(feats, dim=1)
