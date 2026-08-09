"""Short-horizon rolling terrain memory for 11x11 G1 height scans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

from .height_scan_contract import CLIP_MAX, CLIP_MIN, FILL_VALUE, HEIGHT_OFFSET, MAP_SIZE, encode_height, grid_coordinates
from .transforms import Pose2D


class TerrainSource(IntEnum):
    CURRENT_OBSERVATION = 1
    HISTORICAL_MEMORY = 2
    INTERPOLATED = 3
    UNKNOWN = 4


@dataclass
class TerrainCell:
    z: float
    stamp_s: float
    count: int
    confidence: float = 1.0
    source: TerrainSource = TerrainSource.CURRENT_OBSERVATION
    z_min: float = math.nan
    z_max: float = math.nan
    z_span: float = 0.0
    vertical_candidate: bool = False


@dataclass
class TerrainQuery:
    z: float
    confidence: float
    source: TerrainSource
    age_s: float
    radius_cells: int


@dataclass
class HeightScanResult:
    values: np.ndarray
    unknown: np.ndarray
    confidence: np.ndarray
    unknown_ratio: float


class TerrainMemory:
    """Short-horizon elevation memory in a gravity-aligned odometry frame."""

    def __init__(
        self,
        *,
        resolution: float,
        max_age_s: float,
        max_distance: float,
        query_radius_cells: int,
    ) -> None:
        self.resolution = max(1.0e-6, float(resolution))
        self.max_age_s = float(max_age_s)
        self.max_distance = float(max_distance)
        self.query_radius_cells = max(0, int(query_radius_cells))
        self.cells: dict[tuple[int, int], TerrainCell] = {}

    def key(self, world_x: float, world_y: float) -> tuple[int, int]:
        return (
            int(round(float(world_x) / self.resolution)),
            int(round(float(world_y) / self.resolution)),
        )

    def cell_center(self, key: tuple[int, int]) -> tuple[float, float]:
        return key[0] * self.resolution, key[1] * self.resolution

    def update_from_points(
        self,
        points_base: np.ndarray,
        pose: Pose2D,
        stamp_s: float,
        *,
        min_points_per_cell: int,
        percentile: float,
        z_min: float,
        z_max: float,
    ) -> int:
        points = np.asarray(points_base, dtype=np.float32)
        if points.size == 0:
            return 0

        in_height_range = (points[:, 2] > float(z_min)) & (points[:, 2] < float(z_max))
        points = points[in_height_range]
        if points.size == 0:
            return 0

        c = math.cos(pose.yaw)
        s = math.sin(pose.yaw)
        world_x = pose.x + c * points[:, 0] - s * points[:, 1]
        world_y = pose.y + s * points[:, 0] + c * points[:, 1]
        world_z = pose.z + points[:, 2]

        keys = np.rint(np.column_stack((world_x, world_y)) / self.resolution).astype(np.int32)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        updated = 0
        for idx, key_arr in enumerate(unique_keys):
            mask = inverse == idx
            count = int(np.count_nonzero(mask))
            if count < int(min_points_per_cell):
                continue
            key = (int(key_arr[0]), int(key_arr[1]))
            z_values = world_z[mask]
            z_low = float(np.min(z_values))
            z_high = float(np.max(z_values))
            z_span = z_high - z_low
            z = float(np.percentile(z_values, float(percentile)))
            previous = self.cells.get(key)
            if previous is None or stamp_s >= previous.stamp_s:
                self.cells[key] = TerrainCell(
                    z=z,
                    stamp_s=float(stamp_s),
                    count=count,
                    confidence=1.0,
                    source=TerrainSource.CURRENT_OBSERVATION,
                    z_min=z_low,
                    z_max=z_high,
                    z_span=z_span,
                    vertical_candidate=z_span > 0.08,
                )
                updated += 1
        return updated

    def prune(self, pose: Pose2D, now_s: float) -> None:
        if not self.cells:
            return
        max_dist2 = self.max_distance * self.max_distance
        for key, cell in list(self.cells.items()):
            age = float(now_s) - cell.stamp_s
            wx, wy = self.cell_center(key)
            dist2 = (wx - pose.x) ** 2 + (wy - pose.y) ** 2
            if age > self.max_age_s or dist2 > max_dist2:
                del self.cells[key]

    def query(self, world_x: float, world_y: float, now_s: float) -> Optional[TerrainQuery]:
        center = self.key(world_x, world_y)
        best: Optional[tuple[int, float, float, TerrainQuery]] = None
        for dx in range(-self.query_radius_cells, self.query_radius_cells + 1):
            for dy in range(-self.query_radius_cells, self.query_radius_cells + 1):
                key = (center[0] + dx, center[1] + dy)
                cell = self.cells.get(key)
                if cell is None:
                    continue
                age = max(0.0, float(now_s) - cell.stamp_s)
                if age > self.max_age_s:
                    continue

                radius_cells = max(abs(dx), abs(dy))
                dist2 = dx * dx + dy * dy
                if self.max_age_s > 0.0:
                    age_factor = float(np.clip(1.0 - age / self.max_age_s, 0.0, 1.0))
                else:
                    age_factor = 1.0
                confidence = float(np.clip(cell.confidence * age_factor, 0.0, 1.0))
                if radius_cells > 0:
                    confidence = min(confidence, 0.4)
                    source = TerrainSource.INTERPOLATED
                elif age > 1.0e-6:
                    source = TerrainSource.HISTORICAL_MEMORY
                else:
                    source = TerrainSource.CURRENT_OBSERVATION

                query = TerrainQuery(
                    z=cell.z,
                    confidence=confidence,
                    source=source,
                    age_s=age,
                    radius_cells=radius_cells,
                )
                candidate = (dist2, age, -confidence, query)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate

        if best is None:
            return None
        return best[3]

    def query_z(self, world_x: float, world_y: float, now_s: float) -> Optional[float]:
        query = self.query(world_x, world_y, now_s)
        if query is None:
            return None
        return query.z

    def estimate_support_height(
        self,
        pose: Pose2D,
        now_s: float,
        *,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        percentile: float,
    ) -> Optional[float]:
        if not self.cells:
            return None

        c = math.cos(pose.yaw)
        s = math.sin(pose.yaw)
        values: list[float] = []
        for key, cell in self.cells.items():
            if float(now_s) - cell.stamp_s > self.max_age_s:
                continue
            wx, wy = self.cell_center(key)
            dx = wx - pose.x
            dy = wy - pose.y
            local_x = c * dx + s * dy
            local_y = -s * dx + c * dy
            if x_range[0] <= local_x <= x_range[1] and y_range[0] <= local_y <= y_range[1]:
                values.append(cell.z)

        if not values:
            return None
        return float(np.percentile(np.asarray(values, dtype=np.float32), float(percentile)))

    def render_height_scan(
        self,
        pose: Pose2D,
        now_s: float,
        *,
        grid_size: int,
        map_size: float = MAP_SIZE,
        fill_value: float = FILL_VALUE,
        offset: float = HEIGHT_OFFSET,
    ) -> HeightScanResult:
        coords = grid_coordinates(grid_size=int(grid_size), map_size=float(map_size))
        c = math.cos(pose.yaw)
        s = math.sin(pose.yaw)
        terrain_z = np.full(len(coords), np.nan, dtype=np.float32)
        unknown = np.ones(len(coords), dtype=np.uint8)
        confidence = np.zeros(len(coords), dtype=np.float32)

        for idx, (local_x, local_y) in enumerate(coords):
            world_x = pose.x + c * float(local_x) - s * float(local_y)
            world_y = pose.y + s * float(local_x) + c * float(local_y)
            query = self.query(world_x, world_y, now_s)
            if query is None:
                continue
            terrain_z[idx] = np.float32(query.z)
            unknown[idx] = 0
            confidence[idx] = np.float32(query.confidence)

        values = encode_height(
            pose.z,
            terrain_z,
            unknown_mask=unknown.astype(bool),
            fill_value=fill_value,
            offset=offset,
            clip_min=CLIP_MIN,
            clip_max=CLIP_MAX,
        )
        total = max(int(grid_size) * int(grid_size), 1)
        return HeightScanResult(
            values=values,
            unknown=unknown,
            confidence=confidence,
            unknown_ratio=float(np.count_nonzero(unknown)) / float(total),
        )
