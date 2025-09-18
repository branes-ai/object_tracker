import time

import torch


class _Timer:
    def __init__(self, device: torch.device | str | None):
        self.dev = torch.device(device) if device is not None else torch.device("cpu")
        self.cuda = (self.dev.type == "cuda")
        if self.cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
        else:
            self._t0 = 0.0
            self._t1 = 0.0

    def start(self):
        if self.cuda:
            torch.cuda.synchronize(self.dev)
            self._start.record()
        else:
            self._t0 = time.perf_counter()

    def stop_ms(self) -> float:
        if self.cuda:
            self._end.record()
            self._end.synchronize()          # wait
            return float(self._start.elapsed_time(self._end))  # ms
        else:
            self._t1 = time.perf_counter()
            return float((self._t1 - self._t0) * 1000.0)