
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Union
import torch
import numpy as np
from PIL import Image

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
