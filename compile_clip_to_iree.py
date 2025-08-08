import torch, open_clip
import iree.turbine.aot as aot

class VisualWrapper(torch.nn.Module):
    def __init__(self, visual): super().__init__(); self.visual = visual.eval()
    def forward(self, x): return self.visual(x)  # (B,3,224,224)->(B,512)

def main():
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    wrapper = VisualWrapper(model.visual).eval()

    example = torch.randn(2, 3, 224, 224, dtype=torch.float32)  # <-- 224x224, batch=1
    export_out = aot.export(wrapper, example)
    export_out.compile(target_backends=["llvm-cpu"], save_to="clip_vitb32_visual_cpu.vmfb")
    print("Wrote clip_vitb32_visual_cpu.vmfb")

if __name__ == "__main__":
    main()
