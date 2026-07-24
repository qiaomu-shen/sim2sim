#!/usr/bin/env python3
"""Live viewer for the 11x11 height_scan file consumed by the policy."""

from __future__ import annotations

import argparse
import math
import os
import struct
import time
import tkinter as tk
from pathlib import Path


MAGIC = "UTHEIGHT1"


def read_height_scan(path: Path) -> tuple[list[float], int]:
    with path.open("rb") as f:
        header = f.readline().decode("ascii", errors="replace").strip().split()
        if len(header) != 2 or header[0] != MAGIC:
            raise ValueError(f"invalid height scan header in {path}")
        count = int(header[1])
        raw = f.read(count * 4)
        if len(raw) != count * 4:
            raise ValueError(f"short height scan payload in {path}")
    values = list(struct.unpack(f"<{count}f", raw))
    grid_size = int(round(math.sqrt(count)))
    if grid_size * grid_size != count:
        raise ValueError(f"height scan count is not square: {count}")
    return values, grid_size


def read_debug_csv(path: Path, grid_size: int) -> tuple[dict[str, str], list[int] | None]:
    if not path.exists():
        return {}, None

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    meta: dict[str, str] = {}
    unknown: list[int] | None = None
    section = ""
    unknown_rows: list[list[int]] = []

    for line in lines:
        if not line:
            continue
        if line == "values" or line == "unknown":
            section = line
            continue
        if section == "unknown":
            row = [int(float(x)) for x in line.split(",") if x]
            if row:
                unknown_rows.append(row)
            continue
        if section:
            continue
        key, _, value = line.partition(",")
        meta[key] = value

    if len(unknown_rows) == grid_size:
        flat: list[int] = []
        for row in unknown_rows:
            if len(row) != grid_size:
                return meta, None
            flat.extend(row)
        unknown = flat
    return meta, unknown


def color_for(value: float, unknown: bool, flat: float) -> str:
    if unknown:
        return "#5c5c5c"
    norm = max(-1.0, min(1.0, (value - flat) / 0.25))
    if norm < 0.0:
        t = -norm
        r = int(60 + 190 * t)
        g = int(205 - 95 * t)
        b = int(70 - 40 * t)
    else:
        r = int(55 - 20 * norm)
        g = int(205 - 85 * norm)
        b = int(80 + 170 * norm)
    return f"#{r:02x}{g:02x}{b:02x}"


class HeightScanViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.path = Path(args.path)
        self.debug_csv = Path(args.debug_csv) if args.debug_csv else None
        self.cell = args.cell
        self.flat = args.flat_value

        self.root = tk.Tk()
        self.root.title("Estimated height_scan used by policy")
        self.canvas = tk.Canvas(self.root, width=1, height=1, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status = tk.Label(self.root, anchor="w", justify="left", font=("monospace", 10))
        self.status.pack(fill=tk.X)

        self.last_error = ""
        self.root.after(0, self.update)

    def update(self) -> None:
        try:
            values, grid_size = read_height_scan(self.path)
            meta: dict[str, str] = {}
            unknown = None
            if self.debug_csv is not None:
                meta, unknown = read_debug_csv(self.debug_csv, grid_size)
            self.draw(values, grid_size, unknown)
            self.update_status(values, meta)
            self.last_error = ""
        except Exception as exc:  # keep the viewer alive while files appear
            self.last_error = str(exc)
            self.status.config(text=f"waiting for {self.path}: {self.last_error}")
        self.root.after(max(20, int(1000.0 / self.args.rate)), self.update)

    def draw(self, values: list[float], grid_size: int, unknown: list[int] | None) -> None:
        width = grid_size * self.cell
        height = grid_size * self.cell
        self.canvas.config(width=width, height=height)
        self.canvas.delete("all")

        # File order is y-major, x-fast.  Display x-forward upward and y-left leftward.
        for ix in range(grid_size):
            for iy in range(grid_size):
                idx = iy * grid_size + ix
                col = grid_size - 1 - iy
                row = grid_size - 1 - ix
                x0 = col * self.cell
                y0 = row * self.cell
                x1 = x0 + self.cell
                y1 = y0 + self.cell
                is_unknown = unknown is not None and idx < len(unknown) and unknown[idx] != 0
                fill = color_for(values[idx], is_unknown, self.flat)
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#222222")
                self.canvas.create_text(
                    (x0 + x1) * 0.5,
                    (y0 + y1) * 0.5,
                    text=f"{values[idx]:.2f}",
                    fill="#f6f6f6",
                    font=("monospace", max(8, self.cell // 5), "bold"),
                )

        mid = grid_size * self.cell * 0.5
        self.canvas.create_text(mid, 10, text="x forward", fill="#ffffff", font=("monospace", 10))
        self.canvas.create_text(34, mid, text="y left", fill="#ffffff", font=("monospace", 10), angle=90)

    def update_status(self, values: list[float], meta: dict[str, str]) -> None:
        mtime = self.path.stat().st_mtime if self.path.exists() else 0.0
        age_ms = (time.time() - mtime) * 1000.0
        pieces = [
            f"path={self.path}",
            f"age={age_ms:.0f}ms",
            f"min={min(values):.3f}",
            f"max={max(values):.3f}",
        ]
        for key in ("mode", "est_pose", "support_z", "unknown_ratio", "memory_cells"):
            if key in meta:
                pieces.append(f"{key}={meta[key]}")
        self.status.config(text=" | ".join(pieces))

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="/tmp/unitree_g1_height_scan.bin")
    parser.add_argument("--debug-csv", default="/tmp/unitree_g1_height_scan_estimated.csv")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--cell", type=int, default=54)
    parser.add_argument("--flat-value", type=float, default=0.28)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("DISPLAY is not set; run this from a desktop terminal.")
    HeightScanViewer(args).run()


if __name__ == "__main__":
    main()
