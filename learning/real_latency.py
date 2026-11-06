# Checked: CPU latency and size of the fine-tuned networks (laptop). On the robot's CPU, hardware/gonogo/setup_robot.sh measured 34-37 ms per frame for the 5-fold ensemble.
"""python3 -m learning.real_latency"""

import time

import torch

from learning.real_finetune import build

torch.set_num_threads(4)
for arch in ("resnet18", "mobilenet"):
    model = build(arch).eval()
    x = torch.zeros(1, 3, 112, 336)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        start = time.time()
        for _ in range(100):
            model(x)
    print(arch, "params", sum(p.numel() for p in model.parameters()), "ms per frame on 4 threads", round((time.time() - start) * 10, 1), flush=True)
