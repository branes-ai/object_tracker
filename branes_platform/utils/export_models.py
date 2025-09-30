# export_all_models.py
import os
import argparse
from torch import nn
import torch
from torch.export import export, save, Dim

# ----------------------------- Utils ----------------------------- #
def ensure_dirs(root: str):
    od = os.path.join(root, "OD")
    emb = os.path.join(root, "EMB")
    os.makedirs(od, exist_ok=True)
    os.makedirs(emb, exist_ok=True)
    return od, emb

def make_static_example(device, b=1, h=640, w=640, dtype=torch.float32):
    return torch.randn(b, 3, h, w, dtype=dtype, device=device)

def force_math_sdpa():
    # Make attention export-friendly (CLIP/OpenCLIP)
    os.environ["PYTORCH_SDP_FORCE_FALLBACK"] = "1"
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass  # CPU or older torch

# def _freeze_ultralytics_head(model, device, example_hw=(640, 640)):
#     # Warm up to materialize anchors/stride
#     ex = torch.randn(1, 3, example_hw[0], example_hw[1], device=device)
#     with torch.no_grad():
#         _ = model(ex)
#
#     import types
#     for m in model.modules():
#         if hasattr(m, "anchors") and hasattr(m, "stride"):
#             if not isinstance(m.anchors, torch.Tensor):
#                 m.register_buffer("anchors", torch.as_tensor(m.anchors, device=device))
#             if not isinstance(m.stride, torch.Tensor):
#                 m.register_buffer("stride", torch.as_tensor(m.stride, device=device))
#             if hasattr(m, "_inference"):
#                 orig_inf = m._inference
#                 def _inference_noassign(self, x, *args, **kwargs):
#                     # Run original but avoid any attribute reassigns
#                     return orig_inf(x, *args, **kwargs)
#                 m._inference = types.MethodType(_inference_noassign, m)


def _freeze_ultralytics_head(model, device, example_hw=(640, 640)):
    ex = torch.randn(1, 3, example_hw[0], example_hw[1], device=device)
    with torch.no_grad():
        _ = model(ex)
    import types
    for m in model.modules():
        if hasattr(m, "anchors") and hasattr(m, "stride"):
            if not isinstance(m.anchors, torch.Tensor):
                m.register_buffer("anchors", torch.as_tensor(m.anchors, device=device))
            if not isinstance(m.stride, torch.Tensor):
                m.register_buffer("stride", torch.as_tensor(m.stride, device=device))
            if hasattr(m, "_inference"):
                orig_inf = m._inference
                def _inference_noassign(self, x, *args, **kwargs):
                    return orig_inf(x, *args, **kwargs)
                m._inference = types.MethodType(_inference_noassign, m)


# ----------------------- OD model exporters ---------------------- #
def export_yolov8(device: str, out_path: str, yolo_size: int):
    from ultralytics import YOLO
    weight = "yolov8n.pt"
    m = YOLO(weight).to(device)
    m.fuse()
    core = getattr(m, "model", m)
    _freeze_ultralytics_head(core, device, (yolo_size, yolo_size))

    class W(nn.Module):
        def __init__(self, core): super().__init__(); self.core = core
        def forward(self, x): return self.core(x)

    ex = make_static_example(device, 1, yolo_size, yolo_size)
    with torch.no_grad():
        ep = export(W(core).eval(), (ex,), dynamic_shapes=None)
    save(ep, out_path)
    print(f"✅ YOLOv8 -> {out_path} (static {yolo_size}x{yolo_size}, B=1)")

def export_rtdetr(device: str, out_path: str, yolo_size: int):
    from ultralytics import YOLO
    weight = "rtdetr-l.pt"
    m = YOLO(weight).to(device)
    m.fuse()
    core = getattr(m, "model", m)
    _freeze_ultralytics_head(core, device, (yolo_size, yolo_size))

    class W(nn.Module):
        def __init__(self, core): super().__init__(); self.core = core
        def forward(self, x): return self.core(x)

    ex = make_static_example(device, 1, yolo_size, yolo_size)
    try:
        with torch.no_grad():
            ep = export(W(core).eval(), (ex,), dynamic_shapes=None)
        save(ep, out_path)
        print(f"✅ RT-DETR -> {out_path} (static {yolo_size}x{yolo_size}, B=1)")
    except Exception as e:
        print(f"❌ RT-DETR export hit a known channel-mismatch in Ultralytics export path: {e}")

def export_detr(device: str, out_path: str, batch_size: int):
    from transformers import DetrForObjectDetection
    core = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device).eval()

    # Simple static export (avoids modular guards)
    class W(nn.Module):
        def __init__(self, core): super().__init__(); self.core = core
        def forward(self, x):
            out = self.core(pixel_values=x)
            return out.logits, out.pred_boxes

    ex = make_static_example(device, batch_size, 640, 640)
    with torch.no_grad():
        ep = export(W(core).eval(), (ex,), dynamic_shapes=None)
    save(ep, out_path)
    print(f"✅ DETR -> {out_path} (static 640x640, B={batch_size})")

def export_yolos(device: str, out_path: str, batch_size: int):
    from transformers import YolosForObjectDetection
    core = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(device).eval()

    class W(nn.Module):
        def __init__(self, core): super().__init__(); self.core = core
        def forward(self, x):
            out = self.core(pixel_values=x)
            return out.logits, out.pred_boxes

    ex = make_static_example(device, batch_size, 480, 640)
    # H/W must be >=32 to satisfy guards
    ds = {"x": {2: Dim("h", min=32), 3: Dim("w", min=32)}}
    with torch.no_grad():
        ep = export(W(core).eval(), (ex,), dynamic_shapes=ds)
    save(ep, out_path)
    print(f"✅ YOLOS -> {out_path} (dynamic H/W with min=32, B={batch_size})")

def export_fasterrcnn(device: str, out_path: str):
    import torch
    import torch.nn as nn
    from torch.export import export, save
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.image_list import ImageList

    core = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()

    # Freeze the normalization constants into tensors so they’re constants at export time
    img_mean = torch.tensor(core.transform.image_mean, device=device).view(1, 3, 1, 1)
    img_std  = torch.tensor(core.transform.image_std,  device=device).view(1, 3, 1, 1)

    class FixedTransform(nn.Module):
        def __init__(self): super().__init__()
        def forward(self, images, targets=None):
            if isinstance(images, (list, tuple)):
                images = torch.stack(images, dim=0)
            images = (images - img_mean) / img_std
            b, _, h, w = images.shape
            sizes = [(int(h), int(w)) for _ in range(b)]
            return ImageList(images, sizes), targets
        def postprocess(self, result, image_sizes, original_image_sizes):
            return result

    core.transform = FixedTransform()

    # Warm-up to materialize anchors, etc., then freeze anchor mutation
    with torch.no_grad():
        _ = core(torch.randn(1, 3, 800, 800, device=device))
    ag = core.rpn.anchor_generator
    if hasattr(ag, "set_cell_anchors"):
        ag.set_cell_anchors = (lambda *a, **k: None)

    class PreNMS(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            # Transform + backbone
            image_list, _ = self.m.transform(x)
            features = self.m.backbone(image_list.tensors)         # OrderedDict[str, Tensor]
            features_list = list(features.values())                 # <<--- IMPORTANT

            # RPN heads (multi-level lists of tensors)
            objectness, pred_bbox_deltas = self.m.rpn.head(features_list)

            # Anchors need the feature list too
            anchors_per_image = self.m.rpn.anchor_generator(image_list, features_list)
            # anchors_per_image: either List[Tensor] (already cat'd) or List[List[Tensor]]
            apimg0 = anchors_per_image[0]
            if isinstance(apimg0, (list, tuple)):  # older torchvision: per-level tensors
                all_anchors = torch.cat(apimg0, dim=0)  # [A, 4]
            else:  # newer torchvision: already concatenated
                all_anchors = apimg0  # [A, 4]
            # Batch==1 at export; flatten anchors for convenience
            # all_anchors = torch.cat(anchors_per_image[0], dim=0)   # [A,4]

            # Pack feature maps (keep FPN order) + RPN raw heads (concat levels)
            feat_tensors = tuple(features_list)                     # (P2,P3,P4,P5,P6)
            obj_cat = torch.cat([o.flatten(start_dim=2)
                                   .transpose(1,2)
                                   .reshape(-1) for o in objectness], dim=0)        # [A]
            deltas_cat = torch.cat([d.flatten(start_dim=2)
                                      .transpose(1,2)
                                      .reshape(-1, 4) for d in pred_bbox_deltas], dim=0)  # [A,4]

            # Return only tensors (export-friendly)
            return feat_tensors + (all_anchors, obj_cat, deltas_cat)

    ex = torch.randn(1, 3, 800, 800, device=device)
    with torch.no_grad():
        ep = export(PreNMS(core).eval(), (ex,), dynamic_shapes=None)
    save(ep, out_path)
    print(f"✅ Faster R-CNN (pre-NMS heads) -> {out_path} (static 800x800, B=1)")


def export_ssd300(device: str, out_path: str):
    from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

    core = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT).to(device).eval()

    class SSDPreNMS(nn.Module):
        def __init__(self, ssd, device):
            super().__init__()
            self.ssd = ssd
            # Register normalization as buffers to avoid captured free vars
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1))
            self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1))

        def forward(self, x):
            # Expect Bx3x300x300
            x = (x - self.mean) / self.std
            feats = self.ssd.backbone(x)
            # Normalize to list of tensors
            if isinstance(feats, dict):
                feats = list(feats.values())
            elif isinstance(feats, tuple):
                feats = list(feats)
            elif not isinstance(feats, list):
                feats = [feats]
            head_out = self.ssd.head(feats)
            return head_out["bbox_regression"], head_out["cls_logits"]

    wrapped = SSDPreNMS(core, device).eval()
    ex = torch.randn(1, 3, 300, 300, device=device)
    with torch.no_grad():
        ep = torch.export.export(wrapped, (ex,), dynamic_shapes=None)
    torch.export.save(ep, out_path)
    print(f"✅ SSD300 (pre-NMS) -> {out_path} (static 300x300, B=1)")

# -------------------- Embedding model exporters ------------------ #
def export_clip(model_name: str, device: str, out_path: str, batch_size: int):
    import open_clip
    force_math_sdpa()  # make attention export-friendly
    arch = "ViT-B-32" if "b32" in model_name.lower() else "ViT-L-14"
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained="openai")
    visual = model.visual.to(device).eval()

    class W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)

    ex = make_static_example(device, batch_size, 224, 224)
    with torch.no_grad():
        ep = export(W(visual).eval(), (ex,), dynamic_shapes=None)  # static 224x224
    save(ep, out_path)
    print(f"✅ CLIP-{arch} -> {out_path} (static 224x224, B={batch_size})")

def export_dinov2(device: str, out_path: str, batch_size: int):
    from transformers import AutoModel
    core = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()

    class W(nn.Module):
        def __init__(self, core): super().__init__(); self.core = core
        def forward(self, x):
            out = self.core(pixel_values=x)
            return out.last_hidden_state[:, 0, :]  # CLS embedding

    ex = make_static_example(device, batch_size, 224, 224)
    with torch.no_grad():
        ep = export(W(core).eval(), (ex,), dynamic_shapes=None)
    save(ep, out_path)
    print(f"✅ DINOv2-small -> {out_path} (static 224x224, B={batch_size})")

def export_vit_b16(device: str, out_path: str, batch_size: int):
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    m.heads = nn.Identity()
    core = m.to(device).eval()

    class W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)

    ex = make_static_example(device, batch_size, 224, 224)
    with torch.no_grad():
        ep = export(W(core).eval(), (ex,), dynamic_shapes=None)
    save(ep, out_path)
    print(f"✅ ViT-B/16 -> {out_path} (static 224x224, B={batch_size})")

def export_resnet50(device: str, out_path: str, batch_size: int, variant_name: str):
    try:
        import timm
        m = timm.create_model("resnet50", pretrained=True, num_classes=0)
        core = m.to(device).eval()
    except Exception:
        from torchvision.models import resnet50, ResNet50_Weights
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        core = m.to(device).eval()

    class W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)

    ex = make_static_example(device, batch_size, 224, 224)
    with torch.no_grad():
        ep = export(W(core).eval(), (ex,), dynamic_shapes=None)
    save(ep, out_path)
    print(f"✅ {variant_name} -> {out_path} (static 224x224, B={batch_size})")

def export_osnet(arch: str, device: str, out_path: str, batch_size: int):
    try:
        import torchreid
    except Exception as e:
        raise ImportError("OSNet export requires 'torchreid' (pip install torchreid).") from e

    model = torchreid.models.build_model(name=arch, num_classes=0, pretrained=True).to(device).eval()

    class W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)

    ex = make_static_example(device, batch_size, 256, 128)
    with torch.no_grad():
        ep = export(W(model).eval(), (ex,), dynamic_shapes=None)
    save(ep, out_path)
    print(f"✅ OSNet ({arch}) -> {out_path} (static 256x128, B={batch_size})")

def export_mobilenetv2(device: str, out_path: str, batch_size: int):
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier = nn.Identity()
    core = m.to(device).eval()

    class W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)

    ex = make_static_example(device, batch_size, 224, 224)
    with torch.no_grad():
        ep = export(W(core).eval(), (ex,), dynamic_shapes=None)
    save(ep, out_path)
    print(f"✅ MobileNetV2 -> {out_path} (static 224x224, B={batch_size})")

def export_resnet18(device: str, out_path: str, batch_size: int):
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    core = m.to(device).eval()

    class W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)

    ex = make_static_example(device, batch_size, 224, 224)
    with torch.no_grad():
        ep = export(W(core).eval(), (ex,), dynamic_shapes=None)
    save(ep, out_path)
    print(f"✅ ResNet18 -> {out_path} (static 224x224, B={batch_size})")

# ------------------------------- Main ---------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Export OD and embedding models via torch.export")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--export-dir", type=str, default="exports", help="Root export folder")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for export (default=1)")
    parser.add_argument("--yolo-size", type=int, default=640, help="Static size for YOLO/RT-DETR (default=640)")
    parser.add_argument("--models", type=str, nargs="*", default=[
        # OD
        "yolo", "detr", "rt-detr", "yolos",
        "fasterrcnn",
        "ssd300",
        # EMB
        "clip-b32", "clip-l14", "dinov2", "vit_b_16",
        "resnet50", "resnet50_reid",
        "mobilenetv2", "resnet18",
        "osnet_x1_0"
    ], help="List of model names to export")
    args = parser.parse_args()

    od_dir, emb_dir = ensure_dirs(args.export_dir)

    for name in args.models:
        lname = name.lower()
        try:
            # ---------------------- OD ---------------------- #
            if lname == "yolo":
                if args.batch_size != 1:
                    print("ℹ️ YOLOv8 export forces batch_size=1 (overriding).")
                export_yolov8(args.device, os.path.join(od_dir, "yolo.torch_export"), args.yolo_size)

            elif lname == "rt-detr":
                if args.batch_size != 1:
                    print("ℹ️ RT-DETR export forces batch_size=1 (overriding).")
                export_rtdetr(args.device, os.path.join(od_dir, "rt-detr.torch_export"), args.yolo_size)

            elif lname == "detr":
                export_detr(args.device, os.path.join(od_dir, "detr.torch_export"), args.batch_size)

            elif lname == "yolos":
                export_yolos(args.device, os.path.join(od_dir, "yolos.torch_export"), args.batch_size)

            elif lname == "fasterrcnn":
                print("ℹ️ Faster R-CNN export uses static 800x800 and batch_size=1.")
                export_fasterrcnn(args.device, os.path.join(od_dir, "fasterrcnn.torch_export"))

            elif lname == "ssd300":
                print("ℹ️ SSD300 export returns pre-NMS head outputs; static 300x300 and batch_size=1.")
                export_ssd300(args.device, os.path.join(od_dir, "ssd300_preNMS.torch_export"))

            # --------------------- EMB ---------------------- #
            elif lname.startswith("clip"):
                export_clip(lname, args.device, os.path.join(emb_dir, f"{lname}.torch_export"), args.batch_size)

            elif lname == "dinov2":
                export_dinov2(args.device, os.path.join(emb_dir, "dinov2-small.torch_export"), args.batch_size)

            elif lname == "vit_b_16":
                export_vit_b16(args.device, os.path.join(emb_dir, "vit_b_16.torch_export"), args.batch_size)

            elif lname in {"resnet50", "resnet50_reid"}:
                export_resnet50(args.device, os.path.join(emb_dir, f"{lname}.torch_export"), args.batch_size, lname)

            elif lname.startswith("osnet"):
                export_osnet(lname, args.device, os.path.join(emb_dir, f"{lname}.torch_export"), args.batch_size)

            elif lname in {"mobilenetv2", "mobilenet_v2"}:
                export_mobilenetv2(args.device, os.path.join(emb_dir, "mobilenetv2.torch_export"), args.batch_size)

            elif lname in {"resnet18", "resnet-18"}:
                export_resnet18(args.device, os.path.join(emb_dir, "resnet18.torch_export"), args.batch_size)

            else:
                raise ValueError(f"Unknown model name: {name}")

        except Exception as e:
            print(f"❌ Failed to export {name}: {e}")

if __name__ == "__main__":
    main()