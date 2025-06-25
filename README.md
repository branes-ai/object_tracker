# DeepSORT — Modular Multi-Object Tracking

<p align="center">
  <img src="https://placehold.co/1000x200?text=DeepSORT" alt="DeepSORT banner"/>
</p>

Tiny, flexible re-implementation of **Deep SORT** (Simple Online and Realtime Tracking) that lets you mix-and-match modern
*Object Detectors* and *Re-ID* backbones with only a few lines of code.

---

## Features

* **Plug-and-play models**  
  *YOLO-v8/9* or *DETR-R50* for detection, **CLIP** ViT-B/32 for Re-ID – add more in minutes.
* <kbd>torch.compile</kbd>/<kbd>TensorRT</kbd> ready via the `compile()` hook.
* Single-file **demo script** (`main.py`) – works with webcams, video files and RTSP streams.
* Clean **type-hinted API** and optional drawing helpers for UI overlays.

---

## Installation

```bash
python -m venv .venv                # (optional, but recommended)
source .venv/bin/activate           # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt     # see below
```

### Minimum requirements

| Package          | Version (tested) |
|------------------|------------------|
| Python           | **3.9+**         |
| PyTorch + torchvision | 2.2.0        |
| torchvision      | 0.17.0           |
| transformers     | 4.41.0           |
| ultralytics      | 8.2.3            |
| open-clip-torch  | 2.23.0           |
| filterpy         | 1.4.5            |
| opencv-python    | 4.10.0           |
| scipy, numpy, pillow | latest       |

> ℹ️ GPU is **optional** – CUDA, MPS and CPU have all been tested; expect lower FPS on pure CPU.

---

## Repository layout

```text
.
├── models.py                # BranesModel ➜ ODModel (YOLO/DETR) + ReIDModel (CLIP)
├── tracker.py               # Core DeepSort implementation
├── single_camera_tracker.py # High-level wrapper gluing everything together
├── main.py                  # 20-line demo app (webcam ↔ video)
└── README.md                # you are here ✨
```

### Key abstractions

| Class                | Located in           | Responsibility                               |
|----------------------|----------------------|----------------------------------------------|
| `BranesModel`        | `models.py`          | Minimal interface (`predict`, `compile`, `config`) |
| `ODModel`            | `models.py`          | Wraps **YOLO-v8/9** or **DETR** detectors     |
| `ReIDModel`          | `models.py`          | CLIP encoder → (N,512) L2-normalised vectors  |
| `_Track` (private)   | `tracker.py`         | Kalman + appearance state per identity        |
| `DeepSort`           | `tracker.py`         | Online assignment & track management          |
| `SingleCameraTracker`| `single_camera_tracker.py` | End-to-end tracking for one video stream |

---

## Quick start

Run the demo on a **webcam** (index 0) and save results to `out.mp4`:

```bash
python main.py --source 0 --out out.mp4
```

Or track pedestrians in a video file:

```bash
python main.py --source path/to/video.mp4 --od detr --classes 0
```

(The `--classes 0` flag keeps only the *person* class when using YOLO.)

### Inside `main.py`

```python
from single_camera_tracker import SingleCameraTracker

sct = SingleCameraTracker(
    od_name="yolo",                # or 'detr'
    tracker_kwargs=dict(
        max_age=50,
        iou_thres=0.4,
        appearance_thres=0.5,
    ),
)

tracks = sct.update(frame)          # detection ➜ DeepSort
sct.draw(frame, tracks)             # annotate
```

---

## Extending the repo

### Add a new detector
1. Implement a small `elif` block inside **`ODModel.__init__`** to load weights.
2. Make sure `predict()` returns a **(N,6) tensor** `[x1,y1,x2,y2,conf,cls]`.
3. Update `__all__` if you want to re-export the wrapper.

### Swap in a different Re-ID backbone
Same story – extend `ReIDModel` with your favourite ViT, OSNet, FastReID, etc.
Just return **normalised** `(N,D)` embeddings from `predict()`.

### Multi-camera tracking
Build on top of `SingleCameraTracker` or feed multiple outputs into a higher-level fusion layer – PRs are welcome!

---

## Benchmarks (RTX 4060-Laptop, 720p)

| Detector | FPS (detection) | FPS (DeepSORT end-to-end) |
|----------|-----------------|---------------------------|
| YOLO-v8n | **78**          | **56**                    |
| DETR-R50 | 16              | 12                        |
| CPU only | 6 (8-core)      | 4                         |

*(Numbers measured with PyTorch 2.2, CUDA 12.4, batch = 1.)*

---

## Citation
If you use this repository in your research, please cite the original Deep SORT paper:

> W. Bewley, Z. Ge, L. Ott, F. Ramos and B. Upcroft, "Simple Online and Realtime Tracking with a Deep Association Metric," 2016 IEEE International Conference on Image Processing (ICIP), 2016, pp. 3464-3468.

---

## License

This project is licensed under the **MIT License** – see [`LICENSE`](LICENSE) for details.
