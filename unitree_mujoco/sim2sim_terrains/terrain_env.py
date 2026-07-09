#!/usr/bin/env python3
"""Build and launch deterministic G1 sim2sim terrain scenes.

The source definitions live in this directory. Generated MuJoCo scenes are
written beside g1_29dof.xml so all existing mesh paths keep their original
meaning. The stock scene.xml is never modified.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import subprocess
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
UNITREE_MUJOCO = HERE.parent
G1_DIR = UNITREE_MUJOCO / "unitree_robots" / "g1"
DEFAULT_SIMULATOR = UNITREE_MUJOCO / "simulate" / "build" / "unitree_mujoco"
SCENE_PREFIX = "scene_sim2sim_"

FRICTION = "0.8 0.02 0.001"

COLORS = {
  "stairs": "0.28 0.52 0.78 1",
  "stones": "0.78 0.52 0.24 1",
  "rough": "0.38 0.55 0.34 1",
  "gaps": "0.62 0.42 0.68 1",
  "marker": "0.82 0.20 0.18 1",
}

TERRAINS = {
  "flat": "纯平地基准场景",
  "stairs_08": "8 cm 高、30 cm 深的连续上下楼梯",
  "stairs_12": "12 cm 高、30 cm 深的连续上下楼梯",
  "stairs_15": "15 cm 高、30 cm 深的连续上下楼梯",
  "stairs_20": "20 cm 高、30 cm 深的连续上下楼梯",
  "stepping_stones": "高度和横向位置固定可复现的梅花桩路线",
  "rough": "固定随机种子的离散粗糙块地形",
  "gaps": "不同沟宽的连续跨沟测试路线",
  "mixed": "粗糙地形、梅花桩和 15 cm 楼梯串联路线",
}


def _scene_path(name: str) -> Path:
  return G1_DIR / f"{SCENE_PREFIX}{name}.xml"


def _fmt(values: tuple[float, ...] | list[float]) -> str:
  return " ".join(f"{value:.8g}" for value in values)


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
  cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
  cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
  cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
  return (
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  )


def _new_scene(name: str) -> tuple[ET.ElementTree, ET.Element]:
  root = ET.Element("mujoco", {"model": f"g1 sim2sim terrain {name}"})
  ET.SubElement(root, "include", {"file": "g1_29dof.xml"})
  ET.SubElement(root, "statistic", {"center": "4 0 0.5", "extent": "8"})

  visual = ET.SubElement(root, "visual")
  ET.SubElement(
    visual,
    "headlight",
    {
      "diffuse": "0.6 0.6 0.6",
      "ambient": "0.3 0.3 0.3",
      "specular": "0 0 0",
    },
  )
  ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
  ET.SubElement(visual, "global", {"azimuth": "-130", "elevation": "-20"})

  asset = ET.SubElement(root, "asset")
  ET.SubElement(
    asset,
    "texture",
    {
      "type": "skybox",
      "builtin": "gradient",
      "rgb1": "0.3 0.5 0.7",
      "rgb2": "0 0 0",
      "width": "512",
      "height": "3072",
    },
  )
  ET.SubElement(
    asset,
    "texture",
    {
      "type": "2d",
      "name": "groundplane",
      "builtin": "checker",
      "mark": "edge",
      "rgb1": "0.2 0.3 0.4",
      "rgb2": "0.1 0.2 0.3",
      "markrgb": "0.8 0.8 0.8",
      "width": "300",
      "height": "300",
    },
  )
  ET.SubElement(
    asset,
    "material",
    {
      "name": "groundplane",
      "texture": "groundplane",
      "texuniform": "true",
      "texrepeat": "5 5",
      "reflectance": "0.2",
    },
  )

  worldbody = ET.SubElement(root, "worldbody")
  ET.SubElement(
    worldbody,
    "light",
    {"pos": "0 0 3", "dir": "0 0 -1", "directional": "true"},
  )
  ET.SubElement(
    worldbody,
    "geom",
    {
      "name": "floor",
      "type": "plane",
      "size": "0 0 0.05",
      "material": "groundplane",
      "friction": FRICTION,
    },
  )
  return ET.ElementTree(root), worldbody


def _add_box(
  worldbody: ET.Element,
  *,
  name: str,
  pos: tuple[float, float, float],
  full_size: tuple[float, float, float],
  rgba: str,
  euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
  ET.SubElement(
    worldbody,
    "geom",
    {
      "name": name,
      "type": "box",
      "pos": _fmt(pos),
      "size": _fmt(tuple(value / 2.0 for value in full_size)),
      "quat": _fmt(_quat_from_euler(*euler)),
      "rgba": rgba,
      "friction": FRICTION,
      "contype": "1",
      "conaffinity": "1",
    },
  )


def _add_top_box(
  worldbody: ET.Element,
  *,
  name: str,
  center_xy: tuple[float, float],
  full_xy: tuple[float, float],
  top_z: float,
  bottom_z: float = -0.10,
  rgba: str,
  euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
  height = max(0.01, top_z - bottom_z)
  _add_box(
    worldbody,
    name=name,
    pos=(center_xy[0], center_xy[1], bottom_z + height / 2.0),
    full_size=(full_xy[0], full_xy[1], height),
    rgba=rgba,
    euler=euler,
  )


def add_stair_course(
  worldbody: ET.Element,
  *,
  start_x: float,
  center_y: float = 0.0,
  rise: float = 0.15,
  tread: float = 0.30,
  width: float = 1.50,
  steps: int = 6,
  platform_length: float = 1.20,
  prefix: str = "stairs",
) -> float:
  """Add an up-platform-down course and return its ending x coordinate."""
  x = start_x
  for index in range(steps):
    top_z = (index + 1) * rise
    _add_top_box(
      worldbody,
      name=f"{prefix}_up_{index + 1:02d}",
      center_xy=(x + tread / 2.0, center_y),
      full_xy=(tread, width),
      top_z=top_z,
      rgba=COLORS["stairs"],
    )
    x += tread

  platform_top = steps * rise
  _add_top_box(
    worldbody,
    name=f"{prefix}_platform",
    center_xy=(x + platform_length / 2.0, center_y),
    full_xy=(platform_length, width),
    top_z=platform_top,
    rgba=COLORS["stairs"],
  )
  x += platform_length

  for index in range(steps):
    top_z = (steps - index - 1) * rise
    if top_z > 0.0:
      _add_top_box(
        worldbody,
        name=f"{prefix}_down_{index + 1:02d}",
        center_xy=(x + tread / 2.0, center_y),
        full_xy=(tread, width),
        top_z=top_z,
        rgba=COLORS["stairs"],
      )
    x += tread
  return x


def add_stepping_stones(
  worldbody: ET.Element,
  *,
  start_x: float,
  center_y: float = 0.0,
  count: int = 14,
  prefix: str = "stone",
) -> float:
  y_offsets = (0.00, 0.18, -0.16, 0.10, -0.20, 0.04, 0.21, -0.08)
  top_heights = (0.08, 0.12, 0.06, 0.15, 0.10, 0.18, 0.09)
  x = start_x
  for index in range(count):
    x += 0.52
    y = center_y + y_offsets[index % len(y_offsets)]
    top_z = top_heights[index % len(top_heights)]
    size_x = 0.34 + 0.04 * (index % 3)
    size_y = 0.38 + 0.03 * ((index + 1) % 3)
    _add_top_box(
      worldbody,
      name=f"{prefix}_{index + 1:02d}",
      center_xy=(x, y),
      full_xy=(size_x, size_y),
      top_z=top_z,
      rgba=COLORS["stones"],
    )
  return x + 0.35


def add_rough_patch(
  worldbody: ET.Element,
  *,
  start_x: float,
  center_y: float = 0.0,
  rows: int = 7,
  columns: int = 16,
  seed: int = 111,
  prefix: str = "rough",
) -> float:
  rng = random.Random(seed)
  tile_x, tile_y = 0.34, 0.32
  gap = 0.015
  patch_width = rows * (tile_y + gap)
  for column in range(columns):
    for row in range(rows):
      x = start_x + (column + 0.5) * (tile_x + gap)
      y = center_y - patch_width / 2.0 + (row + 0.5) * (tile_y + gap)
      top_z = rng.uniform(0.015, 0.085)
      roll = rng.uniform(-0.045, 0.045)
      pitch = rng.uniform(-0.055, 0.055)
      _add_top_box(
        worldbody,
        name=f"{prefix}_{column:02d}_{row:02d}",
        center_xy=(x, y),
        full_xy=(tile_x, tile_y),
        top_z=top_z,
        rgba=COLORS["rough"],
        euler=(roll, pitch, 0.0),
      )
  return start_x + columns * (tile_x + gap)


def add_gap_course(
  worldbody: ET.Element,
  *,
  start_x: float,
  center_y: float = 0.0,
  width: float = 1.50,
  prefix: str = "gap",
) -> float:
  """Add raised platforms separated by progressively wider gaps."""
  x = start_x
  gap_widths = (0.12, 0.18, 0.24, 0.30, 0.36)
  platform_lengths = (1.00, 0.90, 0.85, 0.80, 1.20)
  for index, (gap, length) in enumerate(zip(gap_widths, platform_lengths)):
    _add_top_box(
      worldbody,
      name=f"{prefix}_platform_{index + 1:02d}",
      center_xy=(x + length / 2.0, center_y),
      full_xy=(length, width),
      top_z=0.08,
      rgba=COLORS["gaps"],
    )
    x += length + gap
    _add_box(
      worldbody,
      name=f"{prefix}_marker_left_{index + 1:02d}",
      pos=(x - gap / 2.0, center_y + width / 2.0 + 0.04, 0.03),
      full_size=(gap, 0.04, 0.06),
      rgba=COLORS["marker"],
    )
    _add_box(
      worldbody,
      name=f"{prefix}_marker_right_{index + 1:02d}",
      pos=(x - gap / 2.0, center_y - width / 2.0 - 0.04, 0.03),
      full_size=(gap, 0.04, 0.06),
      rgba=COLORS["marker"],
    )
  return x


def _populate(name: str, worldbody: ET.Element) -> None:
  if name == "flat":
    return
  if name.startswith("stairs_"):
    rise_cm = int(name.split("_", maxsplit=1)[1])
    add_stair_course(worldbody, start_x=1.50, rise=rise_cm / 100.0)
    return
  if name == "stepping_stones":
    add_stepping_stones(worldbody, start_x=1.20)
    return
  if name == "rough":
    add_rough_patch(worldbody, start_x=1.20)
    return
  if name == "gaps":
    add_gap_course(worldbody, start_x=1.20)
    return
  if name == "mixed":
    x = add_rough_patch(
      worldbody,
      start_x=1.20,
      rows=5,
      columns=10,
      prefix="mixed_rough",
    )
    x = add_stepping_stones(
      worldbody,
      start_x=x + 0.50,
      count=10,
      prefix="mixed_stone",
    )
    add_stair_course(
      worldbody,
      start_x=x + 0.70,
      rise=0.15,
      steps=5,
      prefix="mixed_stairs",
    )
    return
  raise ValueError(f"Unknown terrain: {name}")


def build(name: str) -> Path:
  if name not in TERRAINS:
    available = ", ".join(TERRAINS)
    raise ValueError(f"Unknown terrain '{name}'. Available: {available}")
  tree, worldbody = _new_scene(name)
  _populate(name, worldbody)
  ET.indent(tree, space="  ")
  output_path = _scene_path(name)
  tree.write(output_path, encoding="utf-8", xml_declaration=True)
  return output_path


def _build_all() -> list[Path]:
  return [build(name) for name in TERRAINS]


def _run(name: str, simulator: Path) -> None:
  scene_path = build(name)
  if not simulator.is_file():
    raise FileNotFoundError(
      f"Simulator executable not found: {simulator}. Build unitree_mujoco first."
    )
  subprocess.run([str(simulator), "--scene", str(scene_path)], check=True)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Build or launch selectable G1 sim2sim terrain scenes."
  )
  subparsers = parser.add_subparsers(dest="command", required=True)
  subparsers.add_parser("list", help="List available terrain names.")

  build_parser = subparsers.add_parser("build", help="Build one terrain scene.")
  build_parser.add_argument("name", choices=TERRAINS)
  subparsers.add_parser("build-all", help="Build every terrain scene.")

  run_parser = subparsers.add_parser("run", help="Build and launch one terrain.")
  run_parser.add_argument("name", choices=TERRAINS)
  run_parser.add_argument(
    "--simulator",
    type=Path,
    default=DEFAULT_SIMULATOR,
    help=f"Simulator executable (default: {DEFAULT_SIMULATOR})",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.command == "list":
    for name, description in TERRAINS.items():
      print(f"{name:18s} {description}")
  elif args.command == "build":
    print(build(args.name))
  elif args.command == "build-all":
    for path in _build_all():
      print(path)
  elif args.command == "run":
    _run(args.name, args.simulator.resolve())


if __name__ == "__main__":
  main()
