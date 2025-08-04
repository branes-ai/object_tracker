#!/usr/bin/env python3
"""
evaluate_seg_model.py
Benchmark the unified SegModel (instance/semantic/panoptic) on standard datasets.

Tasks & datasets
----------------
- task=instance  : COCO 2017 val (instances_val2017.json)
                   metrics: bbox mAP (always), mask mAP (optional with --eval-masks & pycocotools)
- task=semantic  : ADE20K val (expects images & label-id PNGs)
                   metrics: mean IoU, pixel accuracy
- task=panoptic  : COCO Panoptic val (panoptic_val2017.json + PNGs)
                   metrics: PQ/SQ/RQ (requires panopticapi)

Usage examples
--------------
# Instance (YOLO-seg) bbox mAP on COCO val, with quick smoke of 100 images
python evaluate_seg_model.py --task instance --backend yolo --coco-root /data/coco \
    --device cpu --max-samples 100 --visualize --vis-num 20

# Instance with mask mAP (needs pycocotools)
python evaluate_seg_model.py --task instance --backend yolo --coco-root /data/coco \
    --eval-masks --device cpu

# Semantic (SegFormer) on ADE20K (expects label-id PNGs)
python evaluate_seg_model.py --task semantic --backend segformer --ade-root /data/ADE20K \
    --device cpu --max-samples 200 --visualize --vis-num 30

# Panoptic (Mask2Former) PQ on COCO Panoptic (needs panopticapi)
python evaluate_seg_model.py --task panoptic --backend mask2former --coco-root /data/coco \
    --coco-panoptic-root /data/coco/annotations \
    --device cpu --max-samples 100

Dependencies
------------
pip install ultralytics transformers torch torchvision torchmetrics pycocotools tqdm
# For panoptic metrics:
pip install git+https://github.com/cocodataset/panopticapi
# For visualization:
pip install opencv-python-headless
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
from tqdm import tqdm

# ------------------ Optional deps & feature flags --------------------- #
_HAS_COCO = False
try:
    from pycocotools import mask as maskUtils  # type: ignore
    _HAS_COCO = True
except Exception:
    pass

_HAS_TM = False
try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision       # instance bbox/mask mAP
    from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy  # semantic
    _HAS_TM = True
except Exception:
    pass

_HAS_PAN = False
try:
    # COCO panoptic API
    from panopticapi.evaluation import pq_compute  # type: ignore
    _HAS_PAN = True
except Exception:
    pass

_HAS_CV2 = False
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    pass

# ----------------------- Project imports (adjust) --------------------- #
from branes_platform.nn.seg.models import SegModel, rles_to_bitmaps  # adjust if your path differs
from branes_platform.nn.seg.visualize import overlay_instances, overlay_semantic, overlay_panoptic


# -------------------------- COCO auto-download ------------------------ #
COCO_VAL_URL     = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNS_URL    = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_PANOPTIC_URL = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"

def download_coco(root: Path, panoptic: bool = False):
    from torchvision.datasets.utils import download_and_extract_archive
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "val2017").exists():
        print("Downloading COCO val2017 images (~1 GB)…")
        download_and_extract_archive(COCO_VAL_URL, download_root=root, extract_root=root)
    anns_root = root / "annotations"
    anns = anns_root / "instances_val2017.json"
    if not anns.exists():
        print("Downloading COCO annotations (~240 MB)…")
        download_and_extract_archive(COCO_ANNS_URL, download_root=root, extract_root=root)
    if panoptic:
        pan_json = anns_root / "panoptic_val2017.json"
        pan_dir  = root / "annotations" / "panoptic_val2017"
        if not pan_json.exists() or not pan_dir.exists():
            print("Downloading COCO panoptic annotations (~820 MB)…")
            download_and_extract_archive(COCO_PANOPTIC_URL, download_root=root, extract_root=root)


# ------------------------------ Helpers ------------------------------- #
def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

def to_bgr_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """torchvision returns CxHxW, float[0,1]. Convert to HxWx3 uint8 (BGR)."""
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return img_np[:, :, ::-1]  # BGR

def coco_anns_to_target_dict(
    coco_target: List[Dict[str, Any]]
) -> Dict[str, torch.Tensor]:
    """GT to TorchMetrics bbox format."""
    boxes: List[List[float]] = []
    labels: List[int] = []
    for obj in coco_target:
        if "bbox" not in obj:
            continue
        x, y, w, h = obj["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(int(obj["category_id"]))
    return {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
    }

def coco_anns_to_mask_tensor(
    coco_target: List[Dict[str, Any]], h: int, w: int
) -> torch.BoolTensor:
    """Rasterize COCO polygon/RLE segmentations to [N,H,W] bool.

    Requires pycocotools. Returns empty tensor if missing or no masks.
    """
    if not _HAS_COCO:
        return torch.zeros((0, h, w), dtype=torch.bool)
    rles = []
    for obj in coco_target:
        seg = obj.get("segmentation", None)
        if seg is None:
            continue
        if isinstance(seg, list):  # polygons
            rle = maskUtils.frPyObjects(seg, h, w)  # type: ignore
            rle = maskUtils.merge(rle)              # merge multiple polygons
        elif isinstance(seg, dict) and {"counts", "size"} <= set(seg.keys()):
            rle = seg
        else:
            continue
        rles.append(rle)
    if not rles:
        return torch.zeros((0, h, w), dtype=torch.bool)
    dec = maskUtils.decode(rles)  # HxWxN or HxW if N==1
    if dec.ndim == 2:
        dec = dec[..., None]
    dec = np.moveaxis(dec, -1, 0).astype(bool)  # N,H,W
    return torch.from_numpy(dec)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def draw_text(img: np.ndarray, text: str, org: Tuple[int, int], color=(255, 255, 255)):
    if not _HAS_CV2:
        return
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ======================= INSTANCE: COCO 2017 val ======================= #
def eval_instance(
    backend: str,
    weight: Optional[str],
    coco_root: Path,
    device: str,
    conf_thres: float,
    max_samples: Optional[int],
    visualize: bool,
    vis_out_dir: Path,
    vis_num: int,
    eval_masks: bool,
    num_workers: int,
):
    assert _HAS_TM, "torchmetrics is required for mAP computation. Install torchmetrics."
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CocoDetection(
        root=coco_root / "val2017",
        annFile=coco_root / "annotations" / "instances_val2017.json",
        transform=transform,
    )
    if max_samples:
        dataset = torch.utils.data.Subset(dataset, range(max_samples))

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=False
    )

    torch_device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    seg = SegModel(task="instance", backend=backend, device=torch_device, weight=weight)

    # mAP for bbox (always) and segm (optional if masks available)
    metric_bbox = MeanAveragePrecision(iou_type="bbox").to(torch_device)
    metric_segm = MeanAveragePrecision(iou_type="segm").to(torch_device) if (eval_masks and _HAS_COCO) else None

    # YOLO label mapping (contiguous) vs COCO category ids
    if backend.startswith("yolo"):
        coco_id2name = {cid: cat["name"] for cid, cat in dataset.dataset.coco.cats.items()}  # type: ignore
        name2coco_id = {v: k for k, v in coco_id2name.items()}
        contig_names = seg.config.get("classes", None)
        if contig_names is None:
            raise RuntimeError("Could not read class names from SegModel.config['classes'] for YOLO mapping.")
        contig2coco_id = [name2coco_id[n] for n in contig_names]
        coco_id2contig = {cid: i for i, cid in enumerate(contig2coco_id)}

        def gt_to_contig(lbls: torch.Tensor) -> torch.Tensor:
            return torch.as_tensor([coco_id2contig[int(i)] for i in lbls], dtype=torch.int64)
        label2name: Any = contig_names
    else:
        gt_to_contig = lambda x: x
        label2name = {cid: cat["name"] for cid, cat in dataset.dataset.coco.cats.items()}  # type: ignore

    vis_count = 0
    if visualize:
        ensure_dir(vis_out_dir)

    pbar = tqdm(loader, desc="Evaluating (instance)", unit="img")
    for imgs, targets in pbar:
        img_t = imgs[0]  # C,H,W
        target = targets[0]  # list[dict]
        H, W = img_t.shape[-2], img_t.shape[-1]
        frame_bgr = to_bgr_uint8(img_t)

        # Predict (request bitmap masks for optional segm mAP)
        res = seg.predict(frame_bgr, conf_thres=conf_thres, mask_format="bitmap")
        preds_dict = {
            "boxes": res.boxes.cpu() if res.boxes is not None else torch.empty((0, 4), dtype=torch.float32),
            "scores": res.scores.cpu() if res.scores is not None else torch.empty((0,), dtype=torch.float32),
            "labels": (res.labels.int().cpu() if res.labels is not None else torch.empty((0,), dtype=torch.int64)),
        }
        if metric_segm is not None and res.masks is not None:
            preds_dict["masks"] = res.masks.cpu()

        # GT conversion
        targets_bbox = coco_anns_to_target_dict(target)
        targets_bbox["labels"] = gt_to_contig(targets_bbox["labels"])

        if metric_segm is not None:
            gt_masks = coco_anns_to_mask_tensor(target, H, W)  # [N,H,W], bool
            # Align GT to GT boxes/labels order:
            # Torchmetrics doesn't require order alignment; it matches by IoU.
            targets_gt = {"masks": gt_masks, "labels": targets_bbox["labels"]}
            metric_segm.update([preds_dict], [targets_gt])

        metric_bbox.update([preds_dict], [targets_bbox])

        # Visualization
        if visualize and vis_count < vis_num:
            vis = overlay_instances(frame_bgr, res, alpha=0.5)
            # Optionally draw GT box count
            if _HAS_CV2:
                draw_text(vis, f"GT: {len(target)}  Pred: {int(res.boxes.shape[0]) if res.boxes is not None else 0}", (10, 20))
            outp = vis_out_dir / f"instance_{vis_count:04d}.jpg"
            import cv2
            cv2.imwrite(str(outp), vis[:, :, ::-1])
            vis_count += 1

    # Results
    print("\n-- COCO-2017 val (Instance) --")
    res_bbox = metric_bbox.compute()
    for k, v in res_bbox.items():
        v = v.cpu()
        print(f"[bbox] {k:<18}: {v.item():.4f}" if v.numel() == 1 else f"[bbox] {k:<18}: {v.tolist()}")

    if metric_segm is not None:
        res_segm = metric_segm.compute()
        for k, v in res_segm.items():
            v = v.cpu()
            print(f"[segm] {k:<18}: {v.item():.4f}" if v.numel() == 1 else f"[segm] {k:<18}: {v.tolist()}")


# ======================== SEMANTIC: ADE20K val ======================== #
class ADE20KDataset(torch.utils.data.Dataset):
    """Minimal ADE20K val loader expecting:
       <ade_root>/images/validation/*.jpg
       <ade_root>/annotations/validation/*.png  (label-id PNG, NOT color-coded)
    """
    def __init__(self, ade_root: Path, transform=None):
        self.img_dir = ade_root / "images" / "validation"
        self.ann_dir = ade_root / "annotations" / "validation"
        self.transform = transform
        assert self.img_dir.exists(), f"Missing {self.img_dir}"
        assert self.ann_dir.exists(), f"Missing {self.ann_dir}"
        # Match by stem
        imgs = sorted(self.img_dir.glob("*.jpg"))
        pngs = {p.stem: p for p in self.ann_dir.glob("*.png")}
        self.items: List[Tuple[Path, Path]] = []
        for im in imgs:
            if im.stem in pngs:
                self.items.append((im, pngs[im.stem]))
        assert len(self.items) > 0, "No paired image/label files found under ADE20K root."

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        import PIL.Image as Image
        img_p, ann_p = self.items[idx]
        img = Image.open(img_p).convert("RGB")
        lab = Image.open(ann_p)  # assume already label indices [0..149], uint8
        if self.transform is not None:
            img = self.transform(img)
        # To tensor HxW long
        lab_np = np.array(lab, dtype=np.int64)
        return img, torch.from_numpy(lab_np)

def eval_semantic(
    backend: str,
    weight: Optional[str],
    ade_root: Path,
    device: str,
    max_samples: Optional[int],
    visualize: bool,
    vis_out_dir: Path,
    vis_num: int,
    num_workers: int,
    num_classes: int,
    ignore_index: Optional[int],
):
    assert _HAS_TM, "torchmetrics is required (MulticlassJaccardIndex / MulticlassAccuracy)."
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ADE20KDataset(ade_root, transform=transform)
    if max_samples:
        dataset = torch.utils.data.Subset(dataset, range(max_samples))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    torch_device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    seg = SegModel(task="semantic", backend=backend, device=torch_device, weight=weight)

    # Metrics
    miou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index).to(torch_device)
    pixacc = MulticlassAccuracy(num_classes=num_classes, ignore_index=ignore_index).to(torch_device)

    vis_count = 0
    if visualize:
        ensure_dir(vis_out_dir)

    pbar = tqdm(loader, desc="Evaluating (semantic)", unit="img")
    for img_t, gt_lab in pbar:
        frame_bgr = to_bgr_uint8(img_t[0])
        H, W = frame_bgr.shape[:2]

        res = seg.predict(frame_bgr)
        pred = res.sem_labels  # [H,W] long on CPU
        assert pred is not None, "Semantic backend returned empty labels."

        # Ensure shapes match
        if pred.shape[0] != H or pred.shape[1] != W:
            # Shouldn't happen (we post-process to input size), but be safe.
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(0).unsqueeze(0).float(), size=(H, W), mode="nearest"
            ).squeeze().to(torch.long)

        miou.update(pred.to(torch_device), gt_lab[0].to(torch_device))
        pixacc.update(pred.to(torch_device), gt_lab[0].to(torch_device))

        if visualize and vis_count < vis_num:
            vis = overlay_semantic(frame_bgr, pred.cpu().numpy(), alpha=0.5)
            if _HAS_CV2:
                draw_text(vis, "Semantic overlay", (10, 20))
            outp = vis_out_dir / f"semantic_{vis_count:04d}.jpg"
            import cv2
            cv2.imwrite(str(outp), vis[:, :, ::-1])
            vis_count += 1

    print("\n-- ADE20K val (Semantic) --")
    print(f"mIoU            : {miou.compute().item():.4f}")
    print(f"Pixel Accuracy  : {pixacc.compute().item():.4f}")


# ====================== PANOPTIC: COCO Panoptic val =================== #
def _write_panoptic_pred_png(pan_map: torch.Tensor, out_path: Path):
    """Save a panoptic map [H,W] int32 as PNG where pixel value encodes segment ID.
    panopticapi expects 24-bit RGB PNG with segment ids; we write uint32 to RGB.
    """
    arr = pan_map.cpu().numpy().astype(np.uint32)
    r = (arr % 256).astype(np.uint8)
    g = ((arr // 256) % 256).astype(np.uint8)
    b = ((arr // 65536) % 256).astype(np.uint8)
    png = np.stack([r, g, b], axis=-1)
    if not _HAS_CV2:
        raise RuntimeError("Saving panoptic PNGs requires OpenCV. Install opencv-python-headless.")
    cv2.imwrite(str(out_path), png[:, :, ::-1])  # write as BGR -> PNG

def eval_panoptic(
    backend: str,
    weight: Optional[str],
    coco_root: Path,
    coco_pan_root: Path,
    device: str,
    max_samples: Optional[int],
    visualize: bool,
    vis_out_dir: Path,
    vis_num: int,
    num_workers: int,
):
    if not _HAS_PAN:
        print("panopticapi not available; skipping PQ/SQ/RQ metrics. Install panopticapi to enable.")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CocoDetection(
        root=coco_root / "val2017",
        annFile=coco_pan_root / "panoptic_val2017.json",
        transform=transform,
    )
    if max_samples:
        dataset = torch.utils.data.Subset(dataset, range(max_samples))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                        num_workers=num_workers)

    torch_device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    seg = SegModel(task="panoptic", backend=backend, device=torch_device, weight=weight)

    vis_count = 0
    if visualize:
        ensure_dir(vis_out_dir)

    # Collect predictions for PQ (if available)
    pred_json: Dict[str, Any] = {
        "annotations": [],
    }
    pred_dir = vis_out_dir / "panoptic_pred_pngs"
    if _HAS_PAN:
        ensure_dir(pred_dir)

    pbar = tqdm(loader, desc="Evaluating (panoptic)", unit="img")
    for imgs, targets in pbar:
        img_t = imgs[0]
        img_id = targets[0][0]["image_id"] if len(targets[0]) > 0 else None
        frame_bgr = to_bgr_uint8(img_t)

        res = seg.predict(frame_bgr, mask_format="rle")  # internal format doesn't matter for PQ
        if visualize and vis_count < vis_num:
            vis = overlay_panoptic(frame_bgr, res, alpha=0.5)
            if _HAS_CV2:
                draw_text(vis, "Panoptic overlay", (10, 20))
            outp = vis_out_dir / f"panoptic_{vis_count:04d}.jpg"
            import cv2
            cv2.imwrite(str(outp), vis[:, :, ::-1])
            vis_count += 1

        if _HAS_PAN and img_id is not None:
            # Save PNG and build segments_info for this image
            png_name = f"{int(img_id):012d}.png"
            _write_panoptic_pred_png(res.panoptic_map, pred_dir / png_name)
            # segments_info should include category_id and id; score is optional
            pred_ann = {
                "image_id": int(img_id),
                "file_name": png_name,
                "segments_info": res.segments_info or [],
            }
            pred_json["annotations"].append(pred_ann)

    if _HAS_PAN:
        # panopticapi expects:
        #  - ground truth JSON (coco_pan_root / panoptic_val2017.json)
        #  - ground truth PNG dir (coco_root / annotations / panoptic_val2017)
        #  - predicted JSON (we pass dict), predicted PNG dir
        gt_json_path = coco_pan_root / "panoptic_val2017.json"
        gt_png_dir   = coco_pan_root / "panoptic_val2017"
        print("\nComputing PQ/SQ/RQ with panopticapi…")
        pq_res = pq_compute(
            gt_json_file=str(gt_json_path),
            gt_folder=str(gt_png_dir),
            pred_json=pred_json,
            pred_folder=str(pred_dir),
        )
        print("\n-- COCO Panoptic val (Panoptic) --")
        # pq_res is a dict with overall and per-category stats
        overall = pq_res["All"]
        print(f"PQ  : {overall['pq']:.4f}  SQ: {overall['sq']:.4f}  RQ: {overall['rq']:.4f}")
        if "Things" in pq_res and "Stuff" in pq_res:
            th = pq_res["Things"]; st = pq_res["Stuff"]
            print(f"PQ_things : {th['pq']:.4f}  PQ_stuff: {st['pq']:.4f}")


# ------------------------------- MAIN --------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SegModel on standard datasets")
    parser.add_argument("--task", required=True, choices=["instance", "semantic", "panoptic"],
                        help="Which segmentation task to evaluate.")
    parser.add_argument("--backend", default=None,
                        help="Backend to use (default per task: yolo | segformer | mask2former)")
    parser.add_argument("--weight", default=None, help="Optional checkpoint override for the backend.")
    parser.add_argument("--device", default="cpu", help="torch device string (e.g., cpu, cuda:0)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of images for a quick smoke test")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")

    # Common datasets roots
    parser.add_argument("--coco-root", default=None, help="Path to COCO root (expects val2017/ and annotations/)")
    parser.add_argument("--coco-panoptic-root", default=None,
                        help="Path to COCO annotations root (contains panoptic_val2017.json and panoptic_val2017/). "
                             "Defaults to <coco-root>/annotations.")
    parser.add_argument("--ade-root", default=None,
                        help="Path to ADE20K root with images/validation and annotations/validation")

    # Instance-specific
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold for instance predictions")
    parser.add_argument("--eval-masks", action="store_true", help="Also compute mask mAP for instance (needs pycocotools)")

    # Download (COCO only)
    parser.add_argument("--download", action="store_true", help="Auto-download COCO val and annotations (and panoptic if task=panoptic)")

    # Visualization
    parser.add_argument("--visualize", action="store_true", help="Save visualization images")
    parser.add_argument("--vis-out-dir", default="./vis_seg", help="Directory to store visualizations")
    parser.add_argument("--vis-num", type=int, default=20, help="Max number of images to visualize")

    # Semantic-specific
    parser.add_argument("--num-classes", type=int, default=150, help="Number of classes for semantic mIoU")
    parser.add_argument("--ignore-index", type=int, default=255, help="Ignore label id for semantic (set to -1 if none)")

    args = parser.parse_args()

    # Resolve defaults
    backend = args.backend
    if backend is None:
        backend = {"instance": "yolo", "semantic": "segformer", "panoptic": "mask2former"}[args.task]

    vis_out_dir = Path(args.vis_out_dir)
    if args.task in ("instance", "panoptic"):
        assert args.coco_root is not None, "--coco-root is required for instance and panoptic tasks."
        coco_root = Path(args.coco_root)
        if args.download:
            download_coco(coco_root, panoptic=(args.task == "panoptic"))
        if args.task == "instance":
            eval_instance(
                backend=backend,
                weight=args.weight,
                coco_root=coco_root,
                device=args.device,
                conf_thres=args.conf_thres,
                max_samples=args.max_samples,
                visualize=args.visualize,
                vis_out_dir=vis_out_dir,
                vis_num=args.vis_num,
                eval_masks=args.eval_masks,
                num_workers=args.num_workers,
            )
        else:
            coco_pan_root = Path(args.coco_panoptic_root) if args.coco_panoptic_root else (coco_root / "annotations")
            eval_panoptic(
                backend=backend,
                weight=args.weight,
                coco_root=coco_root,
                coco_pan_root=coco_pan_root,
                device=args.device,
                max_samples=args.max_samples,
                visualize=args.visualize,
                vis_out_dir=vis_out_dir,
                vis_num=args.vis_num,
                num_workers=args.num_workers,
            )

    elif args.task == "semantic":
        assert args.ade_root is not None, "--ade-root is required for semantic task."
        ade_root = Path(args.ade_root)
        eval_semantic(
            backend=backend,
            weight=args.weight,
            ade_root=ade_root,
            device=args.device,
            max_samples=args.max_samples,
            visualize=args.visualize,
            vis_out_dir=vis_out_dir,
            vis_num=args.vis_num,
            num_workers=args.num_workers,
            num_classes=args.num_classes,
            ignore_index=(None if args.ignore_index < 0 else args.ignore_index),
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()