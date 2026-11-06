# Checked: the exported model matched the PyTorch scores (within 0.02) and ran on the robot.
# Rebuilds gonogo_resnet18.onnx (not in git: 223 MB) from the 5 fold checkpoints. Run from the repo root:
#     python3 hardware/gonogo/export_onnx.py      (needs torch and onnx)
import sys, torch
from pathlib import Path
sys.path.insert(0, '.')
from learning.real_finetune import build
R = Path('runs/real_v1')
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
class Ensemble(torch.nn.Module):
    """uint8 NHWC image in -> mean over folds of (logit - fold threshold). Below the margin = not traversable."""
    def __init__(self, arch):
        super().__init__()
        self.models = torch.nn.ModuleList(); taus = []
        for i in range(5):
            c = torch.load(R / f'finetuned_mask_{arch}_fold{i}.pt', map_location='cpu', weights_only=False)
            m = build(arch); m.load_state_dict(c['state']); m.eval(); self.models.append(m); taus.append(float(c['tau']))
        self.register_buffer('taus', torch.tensor(taus))
    def forward(self, x):
        x = (x.permute(0, 3, 1, 2).float() / 255.0 - MEAN) / STD
        z = torch.stack([m(x).squeeze(1) for m in self.models], 1) - self.taus
        return z.mean(1), z
arch = sys.argv[1]
model = Ensemble(arch).eval()
torch.onnx.export(model, (torch.zeros(1, 112, 336, 3, dtype=torch.uint8),), f'hardware/gonogo/gonogo_{arch}.onnx',
                  input_names=['image'], output_names=['z', 'z_folds'], opset_version=17, dynamo=False)
print('ok', arch)
