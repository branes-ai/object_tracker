# reid_model_turbine.py
import numpy as np, torch, cv2
from typing import Sequence, List, Union
from PIL import Image
import open_clip
from branes_platform.nn.base import BranesModel
from branes_platform.utils.iree import load_vmfb


class ReIDModelTurbine(BranesModel):
    def __init__(self, vmfb="clip_vitb32_224_cpu.vmfb", device="cpu"):
        super().__init__(device)
        self.mod  = load_vmfb(vmfb, device)
        self.enc  = self.mod["forward"]
        _, _, self.prep = open_clip.create_model_and_transforms("ViT-B-32")
        self.D    = 512

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray,
                boxes_xyxy: Union[torch.Tensor, np.ndarray, List[Sequence[float]]]):
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return torch.empty((0, self.D))

        H, W = frame_bgr.shape[:2]
        crops = []
        for x1, y1, x2, y2 in torch.as_tensor(boxes_xyxy).int():
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            if x2 <= x1 or y2 <= y1: continue
            crops.append(Image.fromarray(frame_bgr[y1:y2, x1:x2, ::-1]))

        if not crops:
            return torch.empty((0, self.D))

        batch = torch.stack([self.prep(c) for c in crops]).numpy()
        feats = torch.from_numpy(np.array(self.enc(batch)[0])).float()
        return torch.nn.functional.normalize(feats, dim=1)