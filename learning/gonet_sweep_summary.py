"""Summarise a GONet hyperparameter sweep; choose by VALIDATION accuracy only.

    python3 -m learning.gonet_sweep_summary /path/to/runs/gonet_sweep

Test accuracy is printed for every row for transparency, but the selected
configuration is the one with the best validation accuracy (ties broken by
fewer GAN epochs, then smaller lambda); its test accuracy is the number to
report against the paper's 94.25 %.
"""

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1])
    rows = []
    for config in sorted(root.glob("e*_lam*/stage3_gonet_config.json")):
        result = json.loads(config.read_text())
        epochs, lam = config.parent.name.split("_lam")
        rows.append((int(epochs[1:]), float(lam), result["val_accuracy"], result["test_accuracy"]))
    if not rows:
        sys.exit(f"no finished runs under {root}")
    best = max(rows, key=lambda r: (r[2], -r[0], -r[1]))
    print(f"{'GAN epochs':>10} {'lambda':>7} {'val acc':>8} {'test acc':>9}")
    for row in sorted(rows):
        mark = "  <- selected by validation" if row == best else ""
        print(f"{row[0]:>10} {row[1]:>7} {row[2]:>8.4f} {row[3]:>9.4f}{mark}")
    print(json.dumps({"selected": {"gan_epochs": best[0], "lambda": best[1], "val_accuracy": best[2],
                                   "test_accuracy": best[3]}, "paper_test_accuracy": 0.9425, "runs": len(rows)}))


if __name__ == "__main__":
    main()
