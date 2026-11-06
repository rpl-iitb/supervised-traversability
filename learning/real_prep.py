# Checked by use. Builds the image caches once, before parallel jobs read them.
"""python3 -m learning.real_prep --data DATA --out OUT"""

import argparse
from pathlib import Path

from learning.real_data import build_cache, load_rows
from learning.real_gonet import cache_128

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
rows = load_rows(args.data)
for mode in ("mask", "full"):
    print(mode, build_cache(args.data, rows, mode, args.out / f"images_{mode}.npy").shape, flush=True)
print(cache_128(args.data, rows, args.out / "images_128.npy").shape, flush=True)
