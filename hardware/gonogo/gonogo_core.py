# Checked on the robot, 5-6 Nov 2026 (live trials in the report, Section 5.10).
"""Go/no-go from one camera frame. Shared by the live node (gonogo_node.py) and the bag replay; no ROS imports.

The model is the 5-fold ResNet-18 ensemble trained on session s1 (masked floor corridor input), exported to ONNX.
Each fold's output is its logit minus that fold's validation threshold, so 0 is "on the threshold" for every fold;
``z`` is the mean over folds. Higher = traversable.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
EPISODE_WH = (320, 240)  # training frames were saved at half size
INPUT_WH = (336, 112)


class Model:
    def __init__(self, onnx_path: Path = HERE / "gonogo_resnet18.onnx", mask_path: Path = HERE / "floor_mask.png",
                 threads: int = 4):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        # Idle threads sleep instead of busy-waiting: on the robot, spinning held 3.4 cores at 10 Hz (fan noise,
        # and it competes with the robot's own nodes).
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
        self.mask = np.asarray(Image.open(mask_path)) > 0  # True = floor corridor in front of the robot
        rows = np.flatnonzero(self.mask.any(axis=1))
        self.top, self.bottom = int(rows[0]), int(rows[-1]) + 1

    def preprocess(self, jpeg: bytes) -> tuple[Image.Image, np.ndarray]:
        """The same steps as training: half size, turned upright, saved as JPEG (q92), grey outside the corridor,
        cropped to the corridor rows, resized to 336x112. Returns the upright frame and the model input."""
        image = Image.open(io.BytesIO(jpeg)).convert("RGB").resize(EPISODE_WH).rotate(180)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        image = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
        pixels = np.asarray(image).copy()
        pixels[~self.mask] = 127
        crop = Image.fromarray(pixels[self.top:self.bottom]).resize(INPUT_WH, Image.BILINEAR)
        return image, np.asarray(crop)

    def score(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
        """Returns z (mean over folds), z per fold, and inference time in ms."""
        start = time.perf_counter()
        z, z_folds = self.session.run(None, {"image": np.ascontiguousarray(x[None])})
        return float(z[0]), z_folds[0], (time.perf_counter() - start) * 1000.0


class StopRule:
    """STOP after ``consecutive`` frames in a row with z below ``margin``.

    latch=True (gate): stays stopped until reset(), which the operator triggers with the override button.
    latch=False (shadow): goes back to GO after ``consecutive`` clear frames in a row.
    The defaults (3 frames, margin -1) gave 0 false stops on clean runs and caught every hard crossing on the
    held-out s1 scores; they were picked on those scores, so they are demo settings, not a reported result.
    """

    def __init__(self, margin: float = -1.0, consecutive: int = 3, latch: bool = True):
        self.margin, self.consecutive, self.latch = margin, consecutive, latch
        self.reset()

    def reset(self) -> None:
        self.flagged_run = self.clear_run = 0
        self.stopped = False

    def update(self, z: float) -> bool:
        flagged = z < self.margin
        self.flagged_run = self.flagged_run + 1 if flagged else 0
        self.clear_run = 0 if flagged else self.clear_run + 1
        if self.flagged_run >= self.consecutive:
            self.stopped = True
        elif not self.latch and self.clear_run >= self.consecutive:
            self.stopped = False
        return flagged


def annotate(image: Image.Image, mask: np.ndarray, state: str, z: float) -> bytes:
    """Upright frame, outside of the corridor dimmed, a coloured banner with the decision. JPEG bytes."""
    from PIL import ImageDraw

    pixels = np.asarray(image).copy()
    pixels[~mask] = (pixels[~mask] * 0.45).astype(np.uint8)
    frame = Image.fromarray(pixels).resize((640, 480))
    colour = {"GO": (30, 150, 60), "STOP": (200, 40, 40), "OVERRIDE": (220, 140, 0)}.get(state, (90, 90, 90))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, 640, 56), fill=colour)
    draw.text((16, 12), f"{state}   z = {z:+.2f}", fill=(255, 255, 255), font_size=30)
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()
