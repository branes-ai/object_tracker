from __future__ import annotations
import numpy as np
import cv2
import torch

def depth_to_color(depth_norm: np.ndarray) -> np.ndarray:
    """Map [0,1] depth to a BGR color map (near=warm)."""
    if hasattr(depth_norm, "detach"):
        depth_norm = depth_norm.detach().cpu().numpy()
    depth_u8 = (np.clip(depth_norm, 0, 1) * 255).astype(np.uint8)
    # COLORMAP_INFERNO: good perceptual gradient
    color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    return color  # BGR, uint8

def overlay_depth(frame_bgr: np.ndarray, depth_norm: torch.Tensor | np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color = depth_to_color(depth_norm)
    out = frame_bgr.copy()
    mask = np.ones(frame_bgr.shape[:2], dtype=bool)
    # simple alpha blend
    out[mask] = (out[mask].astype(np.float32)*(1-alpha) + color[mask].astype(np.float32)*alpha).astype(np.uint8)
    return out