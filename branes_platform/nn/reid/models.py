from __future__ import annotations

import warnings
from typing import Any, Dict, List, Sequence, Union
import torch
import numpy as np
from PIL import Image

__all__ = [
    "ReIDModel"
]

from branes_platform.nn.base import BranesModel


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
