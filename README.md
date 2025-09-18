# Object Tracking Benchmark Suite

This repository provides a **unified pipeline** for object tracking
algorithms (DeepSORT, OC-SORT, BoT-SORT, ByteTrack, SORT, …) built on top of
our modular wrappers:

- **ODModel** → popular object detectors (YOLOv8/9, RT-DETR, DETR, YOLOS, Faster R-CNN, SSD, …)  
- **ReIDModel** → appearance encoders for re-identification (CLIP, DINOv2, OSNet, MobileNetV2, ResNet18, …)  
- **Trackers** → multiple multi-object tracking algorithms, pluggable




## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/object-tracking-benchmark.git
cd object-tracking-benchmark

# install dependencies
pip install -r requirements.txt

```

## Download COCO dataset

To run benchmark you first need to download COCO dataset using util script **download_coco.py**

```bash
python -m branes_platform.utils.download_coco --root ./coco
```

## Run benchmark
To run benchmark use the script

```bash
python -m branes_platform.benchmarks.object_trackers.full_object_tracking_benchmark --coco-root ./coco  
```

It benchmarks **FPS vs. number of detected objects** on the
[COCO 2017 validation set](https://cocodataset.org/#download), measuring:

- **OD time (ms)** – detector forward  
- **ReID time (ms)** – embedding extraction  
- **Other tracker Python time (ms)** – association, Kalman filter, etc.  
- **Total time (ms)** – end-to-end update() time  
- **FPS mean**

### Parameters
Check the table below for all available arguments. You can run `python -m branes_platform.benchmarks.object_trackers.full_object_tracking_benchmark --help` to see the same list.

| Argument            | Default      | Description |
|---------------------|--------------|-------------|
| `--od-model`        | `yolo`       | Object detector model. Options: `yolo`, `detr`, `rtdetr`, `yolos`, `fasterrcnn`, `ssd300`. |
| `--reid-model`      | `clip`       | Re-Identification model for appearance features. Examples: `clip`, `clip_vit_b32`, `dinov2_vits14`, `osnet`, `mobilenetv2`, `resnet18`. |
| `--tracker`         | `deep_sort`  | Tracking algorithm. Choices: `deep_sort`, `oc_sort`, `bot_sort`, `bytetrack`, `sort`. |
| `--weight`          | *None*       | Optional path to custom OD checkpoint (e.g., YOLO `.pt` weights). |
| `--coco-root`       | *(required)* | Path to COCO 2017 dataset root (must contain `val2017/` and `annotations/`). |
| `--device`          | `cuda:0`     | Device for inference (`cpu`, `cuda:0`, `mps`, etc.). |
| `--conf-thres`      | `0.25`       | Detection confidence threshold. |
| `--images-per-bin`  | `5`          | How many images to sample per object-count bin. |
| `--repeats`         | `50`         | Number of `update()` calls per image (timed for FPS stats). |
| `--max-objects`     | `20`         | Maximum object count bin to benchmark. |
| `--warmup-iters`    | `3`          | Number of warmup iterations (ignored in results). |
| `--compile-od`      | *flag*       | If set, compile OD model with `torch.compile`. |
| `--compile-reid`    | *flag*       | If set, compile ReID model with `torch.compile`. |
| `--tracker-kwargs`  | `{}`         | Tracker-specific overrides as JSON (e.g., `{"match_iou":0.3}`). |

---