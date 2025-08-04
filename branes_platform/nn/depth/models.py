from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, DPTForDepthEstimation

__all__ = ["DepthModel", "DepthResult"]

@dataclass
class DepthResult:
    # depth_raw: torch.FloatTensor [H, W], arbitrary scale (higher = nearer for DPT hybrid)
    depth_raw: torch.FloatTensor
    # depth_norm: torch.FloatTensor [H, W], min-max normalized to [0,1] (1=near, 0=far)
    depth_norm: torch.FloatTensor
    config: Dict[str, Any]

class DepthModel:
    """Monocular depth via DPT (MiDaS). CPU-friendly default: Intel/dpt-hybrid-midas."""
    def __init__(
        self,
        device: str | torch.device = "cpu",
        weight: str = "Intel/dpt-hybrid-midas",   # good CPU tradeoff
        compile_model: bool = False,
    ) -> None:
        self.device = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(weight)
        self.model = DPTForDepthEstimation.from_pretrained(weight).to(self.device).eval()
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass
        self.config: Dict[str, Any] = {
            "architecture": "DPT",
            "weight": weight,
            "input_format": "BGR uint8 (numpy)",
            "note": "Depth is relative/monotonic, not metric."
        }

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> DepthResult:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
            raise ValueError("frame_bgr must be HxWx3 uint8")
        rgb = frame_bgr[:, :, ::-1]
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        out = self.model(**inputs)

        # Post-process to input size
        # DPT returns inverse depth-like logits in out.predicted_depth
        pred = out.predicted_depth  # [1, H', W']
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(frame_bgr.shape[0], frame_bgr.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)[0]  # [H, W]

        depth_raw = pred.detach().to("cpu").float()
        # Normalize to [0,1] per frame for visualization/fusion
        dmin, dmax = float(depth_raw.min()), float(depth_raw.max())
        depth_norm = (depth_raw - dmin) / max(1e-6, (dmax - dmin))
        return DepthResult(depth_raw=depth_raw, depth_norm=depth_norm, config=self.config.copy())