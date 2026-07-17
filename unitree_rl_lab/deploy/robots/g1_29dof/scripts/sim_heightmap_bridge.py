#!/usr/bin/env python3
"""Write a simulated 121-dim G1 height_scan bridge file for deploy testing."""

from __future__ import annotations

import argparse
import os
import struct
import tempfile
import time
from pathlib import Path


MAGIC = "UTHEIGHT1"


def write_height_file(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{MAGIC} {len(values)}\n".encode("ascii")
    payload += struct.pack(f"<{len(values)}f", *values)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def build_scan(args: argparse.Namespace) -> list[float]:
    n = args.grid_size
    half = args.map_size * 0.5
    step = args.map_size / max(1, n - 1)
    values: list[float] = []

    for iy in range(n):
        y = -half + iy * step
        for ix in range(n):
            x = -half + ix * step
            terrain_h = 0.0
            if args.mode == "step":
                terrain_h = args.height if x >= args.edge_x else 0.0
            elif args.mode == "slope":
                terrain_h = args.height * (x + half) / max(args.map_size, 1e-6)
            elif args.mode == "bump":
                if abs(x - args.center_x) <= args.width * 0.5 and abs(y - args.center_y) <= args.width * 0.5:
                    terrain_h = args.height
            values.append(max(args.clip[0], min(args.clip[1], args.flat_value - terrain_h)))

    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a simulated 11x11 G1 height_scan file for blindwalking deploy."
    )
    parser.add_argument("--output", default="/tmp/unitree_g1_height_scan.bin")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--mode", choices=("flat", "step", "slope", "bump"), default="flat")
    parser.add_argument("--grid-size", type=int, default=11)
    parser.add_argument("--map-size", type=float, default=1.0)
    parser.add_argument(
        "--flat-value",
        type=float,
        default=0.28,
        help="Policy height value for flat ground, roughly pelvis_z - ground_z - 0.5.",
    )
    parser.add_argument("--height", type=float, default=0.10, help="Terrain height in meters.")
    parser.add_argument("--edge-x", type=float, default=0.15, help="Step/slope edge x in current base frame.")
    parser.add_argument("--center-x", type=float, default=0.20)
    parser.add_argument("--center-y", type=float, default=0.0)
    parser.add_argument("--width", type=float, default=0.20)
    parser.add_argument("--clip", type=float, nargs=2, default=[-1.0, 1.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    period = 1.0 / max(args.rate, 1e-6)

    while True:
        values = build_scan(args)
        write_height_file(out, values)
        print(
            f"wrote {out} mode={args.mode} min={min(values):.3f} max={max(values):.3f}",
            flush=True,
        )
        if args.once:
            return
        time.sleep(period)


if __name__ == "__main__":
    main()
