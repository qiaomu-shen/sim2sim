"""ROS-independent heightmap pipeline for G1 LiDAR-memory height scans."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .height_scan_contract import CLIP_MAX, CLIP_MIN, FILL_VALUE, GRID_SIZE, HEIGHT_OFFSET, MAP_SIZE
from .terrain_memory import HeightScanResult, TerrainMemory
from .transforms import (
    MID360_RPY_IN_PELVIS,
    MID360_XYZ_IN_PELVIS,
    Pose2D,
    copy_pose,
    self_filter,
    transform_lidar_to_pelvis,
    transform_past_base_to_current_base,
    voxel_thin,
)


@dataclass
class CloudFrame:
    stamp_s: float
    pose: Pose2D
    points_base: np.ndarray


@dataclass
class HeightMapConfig:
    input_frame: str = "lidar"
    fusion_mode: str = "memory"
    grid_size: int = GRID_SIZE
    map_size: float = MAP_SIZE
    max_frames: int = 8
    max_age: float = 0.8
    voxel: float = 0.02
    min_points_per_cell: int = 1
    percentile: float = 90.0
    fill_value: float = FILL_VALUE
    height_scan_offset: float = HEIGHT_OFFSET
    terrain_z_min: float = -1.50
    terrain_z_max: float = 0.30
    pelvis_height_nominal: float = 0.793
    lidar_xyz: tuple[float, float, float] = MID360_XYZ_IN_PELVIS
    lidar_rpy: tuple[float, float, float] = MID360_RPY_IN_PELVIS
    enable_self_filter: bool = True
    memory_resolution: float = 0.025
    memory_max_age: float = 2.0
    memory_max_distance: float = 2.5
    map_query_radius_cells: int = 3
    disable_base_z_estimator: bool = False
    support_x_range: tuple[float, float] = (-0.24, 0.34)
    support_y_range: tuple[float, float] = (-0.28, 0.28)
    support_percentile: float = 82.0
    base_z_rise_rate: float = 1.10
    base_z_fall_rate: float = 0.45


@dataclass
class HeightMapUpdate:
    result: HeightScanResult
    pose: Pose2D
    point_count: int
    map_updates: int
    memory_cells: int
    frame_count: int
    support_z: float | None


def _empty_result(config: HeightMapConfig) -> HeightScanResult:
    total = int(config.grid_size) * int(config.grid_size)
    return HeightScanResult(
        values=np.full(total, config.fill_value, dtype=np.float32),
        unknown=np.ones(total, dtype=np.uint8),
        confidence=np.zeros(total, dtype=np.float32),
        unknown_ratio=1.0,
    )


def fuse_height_scan(
    frames: deque[CloudFrame],
    current_pose: Pose2D,
    now_s: float,
    *,
    grid_size: int,
    map_size: float,
    max_age_s: float,
    fill_value: float,
    min_points_per_cell: int,
    percentile: float,
    offset: float = HEIGHT_OFFSET,
) -> HeightScanResult:
    terrain_z = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    newest_age = np.full((grid_size, grid_size), np.inf, dtype=np.float32)

    half = map_size * 0.5
    denom = max(grid_size - 1, 1)

    for frame in frames:
        age = float(now_s) - frame.stamp_s
        if age < 0.0 or age > max_age_s:
            continue

        points = transform_past_base_to_current_base(frame.points_base, frame.pose, current_pose)
        if points.size == 0:
            continue

        in_map = (
            (points[:, 0] >= -half)
            & (points[:, 0] <= half)
            & (points[:, 1] >= -half)
            & (points[:, 1] <= half)
            & (points[:, 2] > -1.50)
            & (points[:, 2] < 0.30)
        )
        points = points[in_map]
        if points.size == 0:
            continue

        ix = np.rint((points[:, 0] + half) / map_size * denom).astype(np.int32)
        iy = np.rint((points[:, 1] + half) / map_size * denom).astype(np.int32)
        ix = np.clip(ix, 0, grid_size - 1)
        iy = np.clip(iy, 0, grid_size - 1)

        cell_ids = ix * grid_size + iy
        for cell in np.unique(cell_ids):
            mask = cell_ids == cell
            if np.count_nonzero(mask) < min_points_per_cell:
                continue
            cx = int(cell // grid_size)
            cy = int(cell % grid_size)
            z = float(np.percentile(points[mask, 2], percentile))
            if age <= newest_age[cx, cy]:
                newest_age[cx, cy] = age
                terrain_z[cx, cy] = z

    known = np.isfinite(terrain_z)
    unknown_ratio = 1.0 - float(np.count_nonzero(known)) / float(grid_size * grid_size)

    height_obs = np.full((grid_size, grid_size), fill_value, dtype=np.float32)
    height_obs[known] = -terrain_z[known] - float(offset)
    np.clip(height_obs, CLIP_MIN, CLIP_MAX, out=height_obs)

    confidence = np.zeros((grid_size, grid_size), dtype=np.float32)
    if max_age_s > 0.0:
        confidence[known] = np.clip(1.0 - newest_age[known] / float(max_age_s), 0.0, 1.0)
    else:
        confidence[known] = 1.0

    return HeightScanResult(
        values=height_obs.T.reshape(-1),
        unknown=(~known).T.reshape(-1).astype(np.uint8),
        confidence=confidence.T.reshape(-1),
        unknown_ratio=unknown_ratio,
    )


class HeightMapCore:
    """Pure NumPy heightmap pipeline, independent from ROS message types."""

    def __init__(self, config: HeightMapConfig | None = None) -> None:
        self.config = config or HeightMapConfig()
        self.frames: deque[CloudFrame] = deque(maxlen=self.config.max_frames)
        self.pose = Pose2D(z=self.config.pelvis_height_nominal)
        self.latest_support_height: float | None = None
        self.last_base_z_time: float | None = None
        self.terrain_memory = TerrainMemory(
            resolution=self.config.memory_resolution,
            max_age_s=self.config.memory_max_age,
            max_distance=self.config.memory_max_distance,
            query_radius_cells=self.config.map_query_radius_cells,
        )

    def reset(self) -> None:
        self.frames.clear()
        self.terrain_memory.cells.clear()
        self.pose = Pose2D(z=self.config.pelvis_height_nominal)
        self.latest_support_height = None
        self.last_base_z_time = None

    def add_pose(self, pose: Pose2D, stamp_s: float | None = None) -> None:
        del stamp_s
        self.pose = copy_pose(pose)

    def add_cloud(
        self,
        stamp_s: float,
        points: np.ndarray,
        *,
        now_s: float | None = None,
        input_frame: str | None = None,
    ) -> HeightMapUpdate:
        now = float(stamp_s if now_s is None else now_s)
        frame = input_frame or self.config.input_frame
        points_base = self._prepare_points(points, frame)

        map_updates = 0
        if self.config.fusion_mode == "memory":
            map_updates = self.terrain_memory.update_from_points(
                points_base,
                self.pose,
                float(stamp_s),
                min_points_per_cell=self.config.min_points_per_cell,
                percentile=self.config.percentile,
                z_min=self.config.terrain_z_min,
                z_max=self.config.terrain_z_max,
            )
            self.terrain_memory.prune(self.pose, now)
            self._update_base_z_from_memory(now)
        elif self.config.fusion_mode == "frames":
            self.frames.append(
                CloudFrame(
                    stamp_s=float(stamp_s),
                    pose=copy_pose(self.pose),
                    points_base=points_base,
                )
            )
        else:
            raise ValueError(f"unknown fusion_mode: {self.config.fusion_mode}")

        result = self.render_height_scan(now)
        return HeightMapUpdate(
            result=result,
            pose=copy_pose(self.pose),
            point_count=len(points_base),
            map_updates=map_updates,
            memory_cells=len(self.terrain_memory.cells),
            frame_count=len(self.frames),
            support_z=self.latest_support_height,
        )

    def render_height_scan(self, now_s: float | None = None) -> HeightScanResult:
        now = 0.0 if now_s is None else float(now_s)
        if self.config.fusion_mode == "memory":
            return self.terrain_memory.render_height_scan(
                self.pose,
                now,
                grid_size=self.config.grid_size,
                map_size=self.config.map_size,
                fill_value=self.config.fill_value,
                offset=self.config.height_scan_offset,
            )
        if self.config.fusion_mode == "frames":
            return fuse_height_scan(
                self.frames,
                self.pose,
                now,
                grid_size=self.config.grid_size,
                map_size=self.config.map_size,
                max_age_s=self.config.max_age,
                fill_value=self.config.fill_value,
                min_points_per_cell=self.config.min_points_per_cell,
                percentile=self.config.percentile,
                offset=self.config.height_scan_offset,
            )
        return _empty_result(self.config)

    def _prepare_points(self, points: np.ndarray, input_frame: str) -> np.ndarray:
        cloud = np.asarray(points, dtype=np.float32)
        if cloud.size == 0:
            return cloud.reshape(0, 3)
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise ValueError(f"points must have shape [N, 3], got {cloud.shape}")

        finite = np.isfinite(cloud).all(axis=1)
        cloud = cloud[finite]
        cloud = voxel_thin(cloud, self.config.voxel)
        if input_frame == "lidar":
            cloud = transform_lidar_to_pelvis(cloud, self.config.lidar_xyz, self.config.lidar_rpy)
        elif input_frame in ("base", "pelvis"):
            cloud = cloud
        else:
            raise ValueError(f"unknown input_frame: {input_frame}")
        if self.config.enable_self_filter:
            return self_filter(cloud)
        return cloud

    def _update_base_z_from_memory(self, now_s: float) -> None:
        if self.config.disable_base_z_estimator:
            return

        support_height = self.terrain_memory.estimate_support_height(
            self.pose,
            now_s,
            x_range=self.config.support_x_range,
            y_range=self.config.support_y_range,
            percentile=self.config.support_percentile,
        )
        self.latest_support_height = support_height
        if support_height is None:
            return

        target_z = support_height + self.config.pelvis_height_nominal
        if self.last_base_z_time is None:
            self.last_base_z_time = now_s
            return

        dt = max(0.0, float(now_s) - self.last_base_z_time)
        self.last_base_z_time = now_s
        if dt <= 0.0:
            return

        delta = target_z - self.pose.z
        rate = self.config.base_z_rise_rate if delta > 0.0 else self.config.base_z_fall_rate
        max_delta = max(0.0, rate) * dt
        if max_delta <= 0.0:
            return
        self.pose.z += float(np.clip(delta, -max_delta, max_delta))


def synthetic_stair_points(
    pose: Pose2D,
    *,
    step_x: float,
    step_height: float,
    visible_x_range: tuple[float, float],
    visible_y_range: tuple[float, float],
    resolution: float,
) -> np.ndarray:
    xs = np.arange(visible_x_range[0], visible_x_range[1] + 0.5 * resolution, resolution)
    ys = np.arange(visible_y_range[0], visible_y_range[1] + 0.5 * resolution, resolution)
    points: list[tuple[float, float, float]] = []
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    for local_x in xs:
        for local_y in ys:
            world_x = pose.x + c * local_x - s * local_y
            terrain_z = step_height if world_x >= step_x else 0.0
            points.append((float(local_x), float(local_y), float(terrain_z - pose.z)))
    return np.asarray(points, dtype=np.float32)
