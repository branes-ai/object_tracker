from __future__ import annotations

from typing import List, Sequence, Union, Optional
import numpy as np
import torch
from PIL import Image

from branes_platform.nn.base import BranesModel


class ReIDModelIREE(BranesModel):
    """
    IREE-backed Image Re-Identification encoder (CLIP ViT-B/32 visual).
    Compiled with iree-turbine; default entry is "main".
    """

    def __init__(
        self,
        vmfb_path: str,
        *,
        driver: str = "local-task",     # CPU. Use "cuda"/"vulkan"/"metal" if compiled for those.
        entry: str = "main",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(device)
        self.config.update(
            {
                "architecture": "CLIP-ViT-B/32",
                "pretrained": "laion2b_s34b_b79k",
                "embed_dim": 512,
                "input_size": 224,
            }
        )
        self._driver = driver
        self._entry = entry

        # --- Preprocess (OpenCLIP) ---
        import open_clip
        _, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )

        # --- IREE runtime: VmModule -> SystemContext -> callable attr ---
        import iree.runtime as ireert
        self._cfg = ireert.Config(self._driver)

        with open(vmfb_path, "rb") as f:
            fb = f.read()

        vm_module = ireert.VmModule.from_flatbuffer(self._cfg.vm_instance, fb)

        # Build a context and get the entry ("main" in your case)
        self._ctx = ireert.SystemContext(config=self._cfg, vm_modules=[vm_module])
        mod_ns = self._ctx.modules[vm_module.name]

        if not hasattr(mod_ns, self._entry):
            available = [n for n in dir(mod_ns) if not n.startswith("_")]
            raise RuntimeError(
                f"IREE entry function '{self._entry}' not found. Available: {available}"
            )

        self._fn = getattr(mod_ns, self._entry)

    @torch.no_grad()
    def predict(self, frame_bgr, boxes_xyxy):
        D = 512
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return torch.empty((0, D), dtype=torch.float32, device=self.device)

        H, W, _ = frame_bgr.shape
        boxes = boxes_xyxy.detach().cpu().numpy() if isinstance(boxes_xyxy, torch.Tensor) else np.asarray(boxes_xyxy,
                                                                                                          dtype=np.float32)

        # --- 1) auto-rescale if looks normalized [0,1] ---
        if np.nanmax(boxes) <= 2.0:
            boxes[:, [0, 2]] *= W
            boxes[:, [1, 3]] *= H

        # --- 2) enforce xyxy ordering (fix your x2<x1 / y2<y1) ---
        x1 = np.minimum(boxes[:, 0], boxes[:, 2])
        y1 = np.minimum(boxes[:, 1], boxes[:, 3])
        x2 = np.maximum(boxes[:, 0], boxes[:, 2])
        y2 = np.maximum(boxes[:, 1], boxes[:, 3])
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # --- 3) clip to image bounds ---
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H - 1)

        # --- 4) build crops with min-size gate ---
        min_size = 2  # px
        crops, keep = [], []
        for i, (xa, ya, xb, yb) in enumerate(boxes):
            xa, ya, xb, yb = int(round(xa)), int(round(ya)), int(round(xb)), int(round(yb))
            if (xb - xa) < min_size or (yb - ya) < min_size:
                continue
            crop_rgb = frame_bgr[ya:yb, xa:xb, ::-1]
            if crop_rgb.size == 0:
                continue
            crops.append(Image.fromarray(crop_rgb))
            keep.append(i)

        if not crops:
            # optional debug
            # print("No valid crops. First 3 boxes after sanitize:", boxes[:3])
            return torch.empty((0, D), dtype=torch.float32, device=self.device)

        batch = torch.stack([self.preprocess(img) for img in crops])  # (B,3,224,224)
        batch_np = np.ascontiguousarray(batch.numpy().astype(np.float32))
        # batch_np: (B, 3, 224, 224) float32
        B = batch_np.shape[0]
        BS = 2  # compiled batch size

        pad = (-B) % BS
        if pad:
            # reuse first sample for padding (or zeros — doesn't matter for features we drop)
            pad_block = batch_np[:pad] if B > 0 else np.zeros((pad, 3, 224, 224), np.float32)
            batch_run = np.concatenate([batch_np, pad_block], axis=0)
        else:
            batch_run = batch_np

        outs = []
        for i in range(0, batch_run.shape[0], BS):
            out_i = self._fn(batch_run[i:i + BS])
            if hasattr(out_i, "to_host"):
                out_i = out_i.to_host()
            outs.append(np.asarray(out_i, dtype=np.float32, order="C"))

        feats_np = np.concatenate(outs, axis=0)[:B]  # drop padded rows
        feats = torch.from_numpy(feats_np)
        feats = torch.nn.functional.normalize(feats, dim=1)
        if self.device and str(self.device) != "cpu":
            feats = feats.to(self.device)
        return feats
