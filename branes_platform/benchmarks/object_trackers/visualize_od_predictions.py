#!/usr/bin/env python3
"""
visualize_od_predictions.py
Save sample images with ODModel predictions + COCO ground-truth.

Example
-------
python visualize_od_predictions.py \
    --model yolo \
    --weight yolov8n.pt \
    --coco-root /data/coco \
    --out-dir ./vis \
    --num-images 25 \
    --device cuda:0
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CocoDetection
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#  Adjust this import so that ODModel is importable from your project path.   #
# --------------------------------------------------------------------------- #
from branes_platform.nn.object_detection.models import ODModel     # ← change if you placed the class elsewhere


# --------------------------------------------------------------------------- #
#  Dataset helpers                                                            #
# --------------------------------------------------------------------------- #
def coco_setup(root: Path):
    """Return (dataset, {category_id: name}) for COCO-val2017."""
    transform = transforms.Compose([transforms.ToTensor()])  # RGB float32 [0,1]
    ds = CocoDetection(
        root=root / "val2017",
        annFile=root / "annotations" / "instances_val2017.json",
        transform=transform,
    )
    id2name = {cid: cat["name"] for cid, cat in ds.coco.cats.items()}
    return ds, id2name


def draw_boxes(
    img: np.ndarray,
    boxes: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor | None = None,
    id2name: Dict[int, str] | None = None,
    colour: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> None:
    """Draw boxes (x1,y1,x2,y2) + optional label/confidence on `img` in-place."""
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                      colour, thickness)
        text = str(int(labels[i]))
        if id2name is not None:
            text = id2name.get(int(labels[i]), text)
        if scores is not None:
            text += f" {scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (int(x1), int(y1) - th - 4),
                           (int(x1) + tw, int(y1)), colour, -1)
        cv2.putText(img, text, (int(x1), int(y1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Visualise ODModel predictions")
    ap.add_argument("--model", default="yolo", choices=["yolo", "detr"],
                    help="Detector to wrap (yolo | detr)")
    ap.add_argument("--weight", default=None,
                    help="Optional custom checkpoint path")
    ap.add_argument("--coco-root", required=True,
                    help="Path containing val2017/ and annotations/")
    ap.add_argument("--out-dir", default="./vis",
                    help="Directory to save annotated JPEGs")
    ap.add_argument("--num-images", type=int, default=20,
                    help="How many images to dump")
    ap.add_argument("--conf-thres", type=float, default=0.3,
                    help="Confidence threshold for kept predictions")
    ap.add_argument("--device", default="cuda:0",
                    help="torch device, e.g. cuda:0 or cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- dataset ----------------- #
    print("Loading COCO-val2017 …")
    dataset, id2name = coco_setup(Path(args.coco_root))

    # ----------------- model ------------------- #
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    od = ODModel(model_name=args.model, device=device,
                 **({"weight": args.weight} if args.weight else {}))
    od.model.eval()

    # ----------------- loop -------------------- #
    print(f"Saving {args.num_images} images → {out_dir.resolve()}")
    for idx in tqdm(range(min(args.num_images, len(dataset))), unit="img"):
        img_tensor, target = dataset[idx]
        img_rgb = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = img_rgb[:, :, ::-1]  # BGR for OpenCV / YOLO

        # ---------- predictions ----------
        preds = od.predict(img_bgr, conf_thres=args.conf_thres)
        pred_boxes, pred_scores, pred_labels = preds[:, :4], preds[:, 4], preds[:, 5]

        # ---------- ground-truth ----------
        gt_boxes: List[List[float]] = []
        gt_labels: List[int] = []
        for ann in target:
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(ann["category_id"])
        gt_boxes = np.asarray(gt_boxes, dtype=float)
        gt_labels = np.asarray(gt_labels, dtype=int)

        vis = img_rgb.copy()
        draw_boxes(vis, gt_boxes, gt_labels,
                   id2name=id2name, colour=(0, 255, 0), thickness=2)      # GT = green
        draw_boxes(vis, pred_boxes, pred_labels, pred_scores,
                   id2name=id2name, colour=(0, 0, 255), thickness=2)      # pred = red

        cv2.imwrite(str(out_dir / f"vis_{idx:04d}.jpg"), vis[:, :, ::-1])  # save BGR → JPG

    print("Done!  Open the images in", out_dir)


if __name__ == "__main__":
    main()