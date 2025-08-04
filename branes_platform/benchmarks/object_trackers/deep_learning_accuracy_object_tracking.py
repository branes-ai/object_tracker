#!/usr/bin/env python3
"""evaluate_od_model.py
Evaluate the ODModel wrapper (YOLO‑v8/9 or DETR) on the COCO‑2017 *validation* set.

Usage example
-------------
python evaluate_od_model.py --model yolo --weight yolov8n.pt \
    --coco-root /data/coco --device cuda:0 --batch-size 4

Dependencies
------------
pip install ultralytics transformers torch torchvision torchmetrics pycocotools tqdm
"""

from __future__ import annotations
import cv2
import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

# ----------------------------------------------------------------------------- #
# Adjust this import so that ODModel can be found on your PYTHONPATH            #
# Example: from my_project.models import ODModel                                #
# ----------------------------------------------------------------------------- #
from branes_platform.nn.object_detection.models import ODModel  # <-- EDIT this to match where you placed the class

# --- optional auto-download ---------------------------------------------------
COCO_VAL_URL   = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNS_URL  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

def download_coco(root: Path):
    from torchvision.datasets.utils import download_and_extract_archive
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "val2017").exists():
        print("Downloading COCO val2017 images (~1 GB)…")
        download_and_extract_archive(COCO_VAL_URL, download_root=root, extract_root=root)
    anns = root / "annotations" / "instances_val2017.json"
    if not anns.exists():
        print("Downloading COCO annotations (~240 MB)…")
        download_and_extract_archive(COCO_ANNS_URL, download_root=root, extract_root=root)

def collate_fn(batch):
    """Torch dataloader requires a custom collate for lists of dicts."""
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def coco_target_to_dict(coco_target: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Convert COCO annotation format to TorchMetrics format."""
    boxes: List[List[float]] = []
    labels: List[int] = []
    for obj in coco_target:
        x, y, w, h = obj["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(obj["category_id"])
    return {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
    }
def draw_boxes(img: np.ndarray,
               boxes: np.ndarray,
               labels: np.ndarray,
               scores: np.ndarray | None,
               label2name: list[str] | dict[int, str],
               color: tuple[int, int, int],
               thickness: int = 2) -> None:
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        name = label2name[int(labels[i])] if isinstance(label2name, list) \
               else label2name.get(int(labels[i]), str(int(labels[i])))
        txt = f"{name}"
        if scores is not None:
            txt += f" {float(scores[i]):.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (int(x1), int(y1) - th - 4),
                           (int(x1) + tw, int(y1)), color, -1)
        cv2.putText(img, txt, (int(x1), int(y1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------- #
#  MAIN EVALUATION ROUTINE                                              #
# --------------------------------------------------------------------- #
def run_eval(
    model: str,
    weight: str = None,
    coco_root: str = "./coco",
    batch_size: int = 1,
    device: str = 'cpu',
    conf_thres: float = 0.001,
    max_samples: int = None,
    download: bool = False,
    visualize: bool = False,
    vis_out_dir: str = "./vis",
    vis_num: int = 20,
):
    if download:
        download_coco(Path(coco_root))

    vis_count = 0
    if visualize:
        vis_out = Path(vis_out_dir)
        vis_out.mkdir(parents=True, exist_ok=True)

    # ---------------------------- Dataset ----------------------------- #
    transform = transforms.Compose([transforms.ToTensor()])
    root = Path(coco_root)
    dataset = CocoDetection(
        root=root / "val2017",
        annFile=root / "annotations" / "instances_val2017.json",
        transform=transform,
    )
    if max_samples:
        dataset = torch.utils.data.Subset(dataset, range(max_samples))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,


    )

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model_kwargs = {"weight": weight} if weight else {}
    od_model = ODModel(model_name=model, device=torch_device, **model_kwargs)
    od_model.model.eval()

    metric = MeanAveragePrecision(iou_type="bbox").to(torch_device)

    # ---------- label-ID mapping (YOLO only) --------------------------- #
    if model.startswith("yolo"):
        coco_id2name = {cid: cat["name"] for cid, cat in dataset.coco.cats.items()}
        name2coco_id = {v: k for k, v in coco_id2name.items()}
        contig_id2name = od_model.model.names.values()
        contig2coco_id = [name2coco_id[n] for n in contig_id2name]
        coco_id2contig = {cid: i for i, cid in enumerate(contig2coco_id)}


        def gt_to_contig(lbls: torch.Tensor) -> torch.Tensor:
            return torch.as_tensor([coco_id2contig[int(i)] for i in lbls], dtype=torch.int64)

        label2name = contig_id2name
    else:
        gt_to_contig = lambda x: x
        label2name = {cid: cat["name"] for cid, cat in dataset.coco.cats.items()}

    # ---------------------------- Loop -------------------------------- #
    for imgs, targets in tqdm(dataloader, desc="Evaluating", unit="img"):
        for img_tensor, target in zip(imgs, targets):
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pred = od_model.predict(img_np[:, :, ::-1], conf_thres=conf_thres)

            preds_dict = {
                "boxes": pred[:, :4].cpu(),
                "scores": pred[:, 4].cpu(),
                "labels": pred[:, 5].int().cpu(),
            }
            targets_dict = coco_target_to_dict(target)
            targets_dict["labels"] = gt_to_contig(targets_dict["labels"])
            metric.update([preds_dict], [targets_dict])

            # ---------- visualise ------------------------------------ #
            if visualize and vis_count < vis_num:
                vis_img = img_np.copy()
                draw_boxes(vis_img, targets_dict["boxes"].numpy(),
                           targets_dict["labels"].numpy(), None, label2name, (0, 255, 0))
                draw_boxes(vis_img, preds_dict["boxes"].numpy(),
                           preds_dict["labels"].numpy(), preds_dict["scores"].numpy(),
                           label2name, (0, 0, 255))
                cv2.imwrite(str(vis_out / f"vis_{vis_count:04d}.jpg"), vis_img[:, :, ::-1])
                vis_count += 1
            break
        break

    # --------------------------- Results ------------------------------ #
    print("\n-- COCO-2017 val results --")
    for k, v in metric.compute().items():
        if isinstance(v, torch.Tensor):
            v = v.cpu()
            v = v.item() if v.numel() == 1 else v.tolist()
        print(f"{k:<22}: {v}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ODModel on COCO‑2017 val set")
    parser.add_argument("--model", default="yolo", choices=["yolo", "detr"],
                        help="Which detector to wrap (yolo | detr)")
    parser.add_argument("--weight", default=None,
                        help="Path to custom checkpoint. If omitted, ODModel default is used.")
    parser.add_argument("--coco-root", required=True,
                        help="Path to COCO root containing 'val2017/' and 'annotations/'.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Images per forward pass (keep 1 unless your GPU has lots of RAM)")
    parser.add_argument("--device", default="cuda:0",
                        help="torch device string, e.g. cuda:0 or cpu")
    parser.add_argument("--conf-thres", type=float, default=0.001,
                        help="Confidence threshold for keeping predictions before metric calc")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of images for a quick smoke test")
    parser.add_argument("--download", action="store_true",
                        help="Download COCO‑2017 val set and annotations if not present")
    parser.add_argument("--visualize", action="store_true",
                        help="Save vis-images with GT (green) & predictions (red)")
    parser.add_argument("--vis-out-dir", default="./vis",
                        help="Directory to store visualisations")
    parser.add_argument("--vis-num", type=int, default=20,
                        help="Maximum number of images to visualise")
    args = parser.parse_args()


    run_eval(
        model=args.model,
        weight=args.weight,
        coco_root=args.coco_root,
        batch_size=args.batch_size,
        device=args.device,
        conf_thres=args.conf_thres,
        max_samples=args.max_samples,
        download=args.download,
        visualize=args.visualize,
        vis_out_dir=args.vis_out_dir,
        vis_num=args.vis_num,
    )


if __name__ == "__main__":
    main()