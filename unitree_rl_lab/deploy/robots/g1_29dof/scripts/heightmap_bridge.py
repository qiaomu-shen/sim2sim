#!/usr/bin/env python3

import argparse
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import Imu
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
except ImportError:
    rclpy = None
    Node = object
    Odometry = object
    Imu = object
    PointCloud2 = object
    point_cloud2 = None


MAGIC = "UTHEIGHT1"

# Nominal real-robot G1 Mid360 extrinsic from Unitree's official
# unitree_ros G1 rev_1_0 URDF.  The fixed Mid360 joint is relative to
# torso_link:
#   torso_link -> mid360_link: (0.0002835, 0.00003, 0.428434),
#   rpy=(pi, 0.05112069379091391, 0).
# With the rev_1_0 waist zero pose, pelvis -> torso contributes
# (-0.0039635, 0, 0.044), yielding the pelvis-frame value below.
MID360_XYZ_IN_PELVIS = (-0.00368, 0.00003, 0.472434)
MID360_RPY_IN_PELVIS = (math.pi, 0.05112069379091391, 0.0)


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0


@dataclass
class CloudFrame:
    stamp_s: float
    pose: Pose2D
    points_base: np.ndarray


def yaw_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float32,
    )
    rot_y = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float32,
    )
    rot_z = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return rot_z @ rot_y @ rot_x


def transform_lidar_to_pelvis(
    points_lidar: np.ndarray,
    xyz_in_pelvis: tuple[float, float, float],
    rpy_in_pelvis: tuple[float, float, float],
) -> np.ndarray:
    if points_lidar.size == 0:
        return points_lidar.reshape(0, 3).astype(np.float32, copy=False)
    rot = rotation_from_rpy(*rpy_in_pelvis)
    offset = np.asarray(xyz_in_pelvis, dtype=np.float32)
    return points_lidar @ rot.T + offset


def transform_past_base_to_current_base(
    points_base: np.ndarray,
    frame_pose: Pose2D,
    current_pose: Pose2D,
) -> np.ndarray:
    if points_base.size == 0:
        return points_base.reshape(0, 3).astype(np.float32, copy=False)

    cf = math.cos(frame_pose.yaw)
    sf = math.sin(frame_pose.yaw)
    cc = math.cos(current_pose.yaw)
    sc = math.sin(current_pose.yaw)

    xy = points_base[:, :2]
    world_x = cf * xy[:, 0] - sf * xy[:, 1] + frame_pose.x
    world_y = sf * xy[:, 0] + cf * xy[:, 1] + frame_pose.y

    dx = world_x - current_pose.x
    dy = world_y - current_pose.y

    out = np.empty_like(points_base)
    out[:, 0] = cc * dx + sc * dy
    out[:, 1] = -sc * dx + cc * dy
    out[:, 2] = points_base[:, 2] + frame_pose.z - current_pose.z
    return out


def self_filter(points_base: np.ndarray) -> np.ndarray:
    if points_base.size == 0:
        return points_base.reshape(0, 3)

    x = points_base[:, 0]
    y = points_base[:, 1]
    z = points_base[:, 2]

    torso = (np.abs(x) < 0.34) & (np.abs(y) < 0.30) & (z > -0.18)
    left_leg = (x > -0.26) & (x < 0.22) & (y > 0.035) & (y < 0.22) & (z > -0.82) & (z < -0.10)
    right_leg = (x > -0.26) & (x < 0.22) & (y < -0.035) & (y > -0.22) & (z > -0.82) & (z < -0.10)
    near_feet = (x > -0.12) & (x < 0.18) & (np.abs(y) < 0.22) & (z > -0.88) & (z < -0.64)

    return points_base[~(torso | left_leg | right_leg | near_feet)]


def voxel_thin(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0.0 or len(points) <= 1:
        return points
    keys = np.floor(points / voxel).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


@dataclass
class TerrainCell:
    z: float
    stamp_s: float
    count: int


@dataclass
class HeightScanResult:
    values: np.ndarray
    unknown: np.ndarray
    unknown_ratio: float


class TerrainMemory:
    """Short-horizon elevation memory in the robot's dead-reckoned world frame."""

    def __init__(
        self,
        *,
        resolution: float,
        max_age_s: float,
        max_distance: float,
        query_radius_cells: int,
    ) -> None:
        self.resolution = max(1.0e-6, resolution)
        self.max_age_s = max_age_s
        self.max_distance = max_distance
        self.query_radius_cells = max(0, query_radius_cells)
        self.cells: dict[tuple[int, int], TerrainCell] = {}

    def key(self, world_x: float, world_y: float) -> tuple[int, int]:
        return (
            int(round(world_x / self.resolution)),
            int(round(world_y / self.resolution)),
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
        if points_base.size == 0:
            return 0

        in_height_range = (points_base[:, 2] > z_min) & (points_base[:, 2] < z_max)
        points = points_base[in_height_range]
        if points.size == 0:
            return 0

        c = math.cos(pose.yaw)
        s = math.sin(pose.yaw)
        world_x = pose.x + c * points[:, 0] - s * points[:, 1]
        world_y = pose.y + s * points[:, 0] + c * points[:, 1]
        world_z = pose.z + points[:, 2]

        keys = np.rint(
            np.column_stack((world_x, world_y)) / self.resolution
        ).astype(np.int32)
        updated = 0
        for key_arr in np.unique(keys, axis=0):
            mask = (keys[:, 0] == key_arr[0]) & (keys[:, 1] == key_arr[1])
            count = int(np.count_nonzero(mask))
            if count < min_points_per_cell:
                continue
            key = (int(key_arr[0]), int(key_arr[1]))
            z = float(np.percentile(world_z[mask], percentile))
            previous = self.cells.get(key)
            if previous is None or stamp_s >= previous.stamp_s:
                self.cells[key] = TerrainCell(z=z, stamp_s=stamp_s, count=count)
                updated += 1
        return updated

    def prune(self, pose: Pose2D, now_s: float) -> None:
        if not self.cells:
            return
        max_dist2 = self.max_distance * self.max_distance
        for key, cell in list(self.cells.items()):
            age = now_s - cell.stamp_s
            wx, wy = self.cell_center(key)
            dist2 = (wx - pose.x) ** 2 + (wy - pose.y) ** 2
            if age > self.max_age_s or dist2 > max_dist2:
                del self.cells[key]

    def query_z(self, world_x: float, world_y: float, now_s: float) -> Optional[float]:
        center = self.key(world_x, world_y)
        best: Optional[tuple[int, float, float]] = None
        for dx in range(-self.query_radius_cells, self.query_radius_cells + 1):
            for dy in range(-self.query_radius_cells, self.query_radius_cells + 1):
                key = (center[0] + dx, center[1] + dy)
                cell = self.cells.get(key)
                if cell is None or now_s - cell.stamp_s > self.max_age_s:
                    continue
                dist = dx * dx + dy * dy
                age = now_s - cell.stamp_s
                candidate = (dist, age, cell.z)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            return None
        return best[2]

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
            if now_s - cell.stamp_s > self.max_age_s:
                continue
            wx, wy = self.cell_center(key)
            dx = wx - pose.x
            dy = wy - pose.y
            local_x = c * dx + s * dy
            local_y = -s * dx + c * dy
            if (
                x_range[0] <= local_x <= x_range[1]
                and y_range[0] <= local_y <= y_range[1]
            ):
                values.append(cell.z)

        if not values:
            return None
        return float(np.percentile(np.asarray(values, dtype=np.float32), percentile))

    def render_height_scan(
        self,
        pose: Pose2D,
        now_s: float,
        *,
        grid_size: int,
        map_size: float,
        fill_value: float,
        offset: float,
    ) -> HeightScanResult:
        half = map_size * 0.5
        denom = max(grid_size - 1, 1)
        c = math.cos(pose.yaw)
        s = math.sin(pose.yaw)
        values: list[float] = []
        unknown_mask: list[int] = []
        unknown = 0

        for iy in range(grid_size):
            local_y = -half + map_size * float(iy) / float(denom)
            for ix in range(grid_size):
                local_x = -half + map_size * float(ix) / float(denom)
                world_x = pose.x + c * local_x - s * local_y
                world_y = pose.y + s * local_x + c * local_y
                terrain_z = self.query_z(world_x, world_y, now_s)
                if terrain_z is None:
                    values.append(fill_value)
                    unknown_mask.append(1)
                    unknown += 1
                    continue
                value = pose.z - terrain_z - offset
                values.append(float(np.clip(value, -1.0, 1.0)))
                unknown_mask.append(0)

        total = max(grid_size * grid_size, 1)
        return HeightScanResult(
            values=np.asarray(values, dtype=np.float32),
            unknown=np.asarray(unknown_mask, dtype=np.uint8),
            unknown_ratio=float(unknown) / float(total),
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
) -> HeightScanResult:
    terrain_z = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    newest_age = np.full((grid_size, grid_size), np.inf, dtype=np.float32)

    half = map_size * 0.5
    denom = max(grid_size - 1, 1)

    for frame in frames:
        age = now_s - frame.stamp_s
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
    # points are in current pelvis/base frame, so terrain_z is terrain height
    # relative to pelvis.  Training target is pelvis_z - terrain_z - 0.5.
    height_obs[known] = -terrain_z[known] - 0.5
    np.clip(height_obs, -1.0, 1.0, out=height_obs)
    # IsaacLab grid_pattern(ordering="xy") keeps x as the fast axis.
    return HeightScanResult(
        values=height_obs.T.reshape(-1),
        unknown=(~known).T.reshape(-1).astype(np.uint8),
        unknown_ratio=unknown_ratio,
    )


def write_height_file(path: str, values: np.ndarray) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(f"{MAGIC} {len(values)}\n".encode("ascii"))
        f.write(np.asarray(values, dtype=np.float32).tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def write_debug_csv(
    path: str,
    values: np.ndarray,
    unknown: np.ndarray,
    *,
    mode: str,
    pose: Pose2D,
    unknown_ratio: float,
    memory_cells: int,
    map_updates: int,
    point_count: int,
    support_z: Optional[float],
) -> None:
    if not path:
        return

    grid_size = int(round(math.sqrt(len(values))))
    if grid_size * grid_size != len(values):
        return

    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    support_text = "nan" if support_z is None else f"{support_z:.6f}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(f"mode,{mode}\n")
        f.write(f"est_pose,{pose.x:.6f} {pose.y:.6f} {pose.z:.6f} {pose.yaw:.6f}\n")
        f.write(f"support_z,{support_text}\n")
        f.write(f"unknown_ratio,{unknown_ratio:.6f}\n")
        f.write(f"memory_cells,{memory_cells}\n")
        f.write(f"map_updates,{map_updates}\n")
        f.write(f"point_count,{point_count}\n")
        f.write("values\n")
        for iy in range(grid_size):
            row = values[iy * grid_size : (iy + 1) * grid_size]
            f.write(",".join(f"{float(v):.6f}" for v in row))
            f.write("\n")
        f.write("unknown\n")
        for iy in range(grid_size):
            row = unknown[iy * grid_size : (iy + 1) * grid_size]
            f.write(",".join(str(int(v)) for v in row))
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


class HeightMapBridge(Node):
    def __init__(self, args):
        super().__init__("unitree_g1_heightmap_bridge")
        self.args = args
        self.frames: deque[CloudFrame] = deque(maxlen=args.max_frames)
        self.pose = Pose2D(z=args.pelvis_height_nominal)
        self.have_odom = args.motion_source != "odom" or not bool(args.odom_topic)
        self.last_warn_time = 0.0
        self.last_info_time = 0.0
        self.last_motion_time: Optional[float] = None
        self.last_base_z_time: Optional[float] = None
        self.imu_yaw_rate = 0.0
        self.latest_support_height: Optional[float] = None

        self.lidar_xyz = tuple(args.lidar_xyz)
        self.lidar_rpy = tuple(args.lidar_rpy)
        if args.lidar_pitch is not None:
            self.lidar_rpy = (0.0, args.lidar_pitch, 0.0)
        self.terrain_memory = TerrainMemory(
            resolution=args.memory_resolution,
            max_age_s=args.memory_max_age,
            max_distance=args.memory_max_distance,
            query_radius_cells=args.map_query_radius_cells,
        )

        self.cloud_sub = self.create_subscription(
            PointCloud2, args.pointcloud_topic, self.on_cloud, 5
        )
        self.odom_sub = None
        if args.motion_source == "odom" and args.odom_topic:
            self.odom_sub = self.create_subscription(
                Odometry, args.odom_topic, self.on_odom, 20
            )
        self.imu_sub = None
        if args.motion_source == "dead-reckon" and args.imu_topic:
            self.imu_sub = self.create_subscription(Imu, args.imu_topic, self.on_imu, 50)

        self.get_logger().info(
            "bridging %s -> %s, fusion=%s, motion=%s, mid360 xyz in pelvis=(%.5f, %.5f, %.5f), rpy=(%.5f, %.5f, %.5f) rad"
            % (
                args.pointcloud_topic,
                args.output,
                args.fusion_mode,
                args.motion_source,
                self.lidar_xyz[0],
                self.lidar_xyz[1],
                self.lidar_xyz[2],
                self.lidar_rpy[0],
                self.lidar_rpy[1],
                self.lidar_rpy[2],
            )
        )

    def update_dead_reckon(self, now_s: float) -> None:
        if self.args.motion_source != "dead-reckon":
            return
        if self.last_motion_time is None:
            self.last_motion_time = now_s
            return

        dt = now_s - self.last_motion_time
        if dt <= 0.0:
            return
        dt = min(dt, self.args.dead_reckon_max_dt)
        self.last_motion_time = now_s

        if self.args.use_imu_yaw_rate:
            self.pose.yaw += self.imu_yaw_rate * dt
        vx = self.args.dead_reckon_vx
        vy = self.args.dead_reckon_vy
        c = math.cos(self.pose.yaw)
        s = math.sin(self.pose.yaw)
        self.pose.x += (c * vx - s * vy) * dt
        self.pose.y += (s * vx + c * vy) * dt

    def update_base_z_from_memory(self, now_s: float) -> None:
        if self.args.disable_base_z_estimator:
            return

        support_height = self.terrain_memory.estimate_support_height(
            self.pose,
            now_s,
            x_range=tuple(self.args.support_x_range),
            y_range=tuple(self.args.support_y_range),
            percentile=self.args.support_percentile,
        )
        self.latest_support_height = support_height
        if support_height is None:
            return

        target_z = support_height + self.args.pelvis_height_nominal
        if self.last_base_z_time is None:
            self.last_base_z_time = now_s
            return

        dt = max(0.0, now_s - self.last_base_z_time)
        self.last_base_z_time = now_s
        if dt <= 0.0:
            return

        delta = target_z - self.pose.z
        rate = self.args.base_z_rise_rate if delta > 0.0 else self.args.base_z_fall_rate
        max_delta = max(0.0, rate) * dt
        if max_delta <= 0.0:
            return
        delta = float(np.clip(delta, -max_delta, max_delta))
        self.pose.z += delta

    def on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        self.pose = Pose2D(
            x=float(p.x),
            y=float(p.y),
            z=float(p.z),
            yaw=yaw_from_quat_xyzw(float(q.x), float(q.y), float(q.z), float(q.w)),
        )
        self.have_odom = True

    def on_imu(self, msg: Imu) -> None:
        self.imu_yaw_rate = float(msg.angular_velocity.z)

    def on_cloud(self, msg: PointCloud2) -> None:
        now_s = self.get_clock().now().nanoseconds * 1.0e-9
        self.update_dead_reckon(now_s)
        if not self.have_odom and len(self.frames) > 0:
            self.throttled_warn(
                f"no odometry received on {self.args.odom_topic}; LiDAR history is not motion-compensated yet"
            )

        stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1.0e-9
        if stamp_s <= 0.0:
            stamp_s = now_s

        points = self.read_xyz(msg)
        if points.size == 0:
            self.throttled_warn("skip cloud: no finite xyz points")
            return

        points = voxel_thin(points, self.args.voxel)
        if self.args.input_frame == "lidar":
            points_base = transform_lidar_to_pelvis(points, self.lidar_xyz, self.lidar_rpy)
        else:
            points_base = points

        points_base = self_filter(points_base)
        map_updates = 0
        if self.args.fusion_mode == "memory":
            map_updates = self.terrain_memory.update_from_points(
                points_base,
                self.pose,
                stamp_s,
                min_points_per_cell=self.args.min_points_per_cell,
                percentile=self.args.percentile,
                z_min=self.args.terrain_z_min,
                z_max=self.args.terrain_z_max,
            )
            self.terrain_memory.prune(self.pose, now_s)
            self.update_base_z_from_memory(now_s)
            result = self.terrain_memory.render_height_scan(
                self.pose,
                now_s,
                grid_size=self.args.grid_size,
                map_size=self.args.map_size,
                fill_value=self.args.fill_value,
                offset=self.args.height_scan_offset,
            )
        else:
            self.frames.append(CloudFrame(stamp_s=stamp_s, pose=self.pose, points_base=points_base))
            result = fuse_height_scan(
                self.frames,
                self.pose,
                now_s,
                grid_size=self.args.grid_size,
                map_size=self.args.map_size,
                max_age_s=self.args.max_age,
                fill_value=self.args.fill_value,
                min_points_per_cell=self.args.min_points_per_cell,
                percentile=self.args.percentile,
            )
        height_scan = result.values
        unknown_ratio = result.unknown_ratio
        write_height_file(self.args.output, height_scan)
        write_debug_csv(
            self.args.debug_csv,
            height_scan,
            result.unknown,
            mode=self.args.fusion_mode,
            pose=self.pose,
            unknown_ratio=unknown_ratio,
            memory_cells=len(self.terrain_memory.cells),
            map_updates=map_updates,
            point_count=len(points_base),
            support_z=self.latest_support_height,
        )

        if now_s - self.last_info_time > self.args.log_interval:
            self.get_logger().info(
                "height_scan min=%.3f max=%.3f unknown=%.1f%% frames=%d points=%d map_cells=%d map_updates=%d pose=(%.3f, %.3f, %.3f, %.3f) support_z=%s"
                % (
                    float(np.min(height_scan)),
                    float(np.max(height_scan)),
                    unknown_ratio * 100.0,
                    len(self.frames),
                    len(points_base),
                    len(self.terrain_memory.cells),
                    map_updates,
                    self.pose.x,
                    self.pose.y,
                    self.pose.z,
                    self.pose.yaw,
                    "nan" if self.latest_support_height is None else f"{self.latest_support_height:.3f}",
                )
            )
            self.last_info_time = now_s

    def read_xyz(self, msg: PointCloud2) -> np.ndarray:
        rows = point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        if isinstance(rows, np.ndarray) and rows.dtype.fields:
            points = np.column_stack((rows["x"], rows["y"], rows["z"])).astype(np.float32)
        else:
            points = np.asarray(list(rows), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            return np.empty((0, 3), dtype=np.float32)
        finite = np.isfinite(points).all(axis=1)
        return points[finite]

    def throttled_warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self.last_warn_time > 2.0:
            self.get_logger().warn(message)
            self.last_warn_time = now


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Bridge a ROS2 PointCloud2 LiDAR topic to the 121-dim Unitree G1 "
            "height_scan bridge file expected by the 29-DOF policy."
        )
    )
    parser.add_argument("--pointcloud-topic", default="/livox/lidar")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--imu-topic", default="/imu")
    parser.add_argument("--output", default="/tmp/unitree_g1_height_scan.bin")
    parser.add_argument("--debug-csv", default="")
    parser.add_argument("--input-frame", choices=("lidar", "base"), default="lidar")
    parser.add_argument("--fusion-mode", choices=("frames", "memory"), default="frames")
    parser.add_argument("--motion-source", choices=("odom", "dead-reckon", "none"), default="odom")
    parser.add_argument("--grid-size", type=int, default=11)
    parser.add_argument("--map-size", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-age", type=float, default=0.8)
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--min-points-per-cell", type=int, default=1)
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--fill-value", type=float, default=0.28)
    parser.add_argument("--write-initial-flat", action="store_true")
    parser.add_argument("--height-scan-offset", type=float, default=0.5)
    parser.add_argument("--terrain-z-min", type=float, default=-1.50)
    parser.add_argument("--terrain-z-max", type=float, default=0.30)
    parser.add_argument("--pelvis-height-nominal", type=float, default=0.793)
    parser.add_argument("--dead-reckon-vx", type=float, default=0.6)
    parser.add_argument("--dead-reckon-vy", type=float, default=0.0)
    parser.add_argument("--dead-reckon-max-dt", type=float, default=0.05)
    parser.add_argument("--use-imu-yaw-rate", action="store_true")
    parser.add_argument("--memory-resolution", type=float, default=0.05)
    parser.add_argument("--memory-max-age", type=float, default=1.4)
    parser.add_argument("--memory-max-distance", type=float, default=2.0)
    parser.add_argument("--map-query-radius-cells", type=int, default=1)
    parser.add_argument("--disable-base-z-estimator", action="store_true")
    parser.add_argument("--support-x-range", type=float, nargs=2, default=[-0.20, 0.28])
    parser.add_argument("--support-y-range", type=float, nargs=2, default=[-0.24, 0.24])
    parser.add_argument("--support-percentile", type=float, default=75.0)
    parser.add_argument("--base-z-rise-rate", type=float, default=0.80)
    parser.add_argument("--base-z-fall-rate", type=float, default=0.35)
    parser.add_argument(
        "--self-test-stair",
        action="store_true",
        help="Run a no-ROS synthetic stair memory test and exit.",
    )
    parser.add_argument(
        "--lidar-xyz",
        type=float,
        nargs=3,
        default=list(MID360_XYZ_IN_PELVIS),
        metavar=("X", "Y", "Z"),
        help="mid360 origin in pelvis frame, meters.",
    )
    parser.add_argument(
        "--lidar-rpy",
        type=float,
        nargs=3,
        default=list(MID360_RPY_IN_PELVIS),
        metavar=("ROLL", "PITCH", "YAW"),
        help="mid360 frame rotation in pelvis frame, radians.",
    )
    parser.add_argument(
        "--lidar-pitch",
        type=float,
        default=None,
        help="legacy pitch-only override; sets lidar rpy to (0, pitch, 0).",
    )
    parser.add_argument("--log-interval", type=float, default=1.0)
    return parser.parse_known_args(argv)


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


def apply_base_z_target(
    pose: Pose2D,
    target_z: float,
    dt: float,
    *,
    rise_rate: float,
    fall_rate: float,
) -> None:
    delta = target_z - pose.z
    rate = rise_rate if delta > 0.0 else fall_rate
    max_delta = max(0.0, rate) * max(0.0, dt)
    if max_delta <= 0.0:
        return
    pose.z += float(np.clip(delta, -max_delta, max_delta))


def run_synthetic_stair_test(args) -> int:
    memory = TerrainMemory(
        resolution=args.memory_resolution,
        max_age_s=args.memory_max_age,
        max_distance=args.memory_max_distance,
        query_radius_cells=args.map_query_radius_cells,
    )
    pose = Pose2D(z=args.pelvis_height_nominal)
    dt = 0.05
    step_x = 0.45
    step_height = 0.15
    max_pose_z = pose.z
    final_unknown_ratio = 1.0

    for i in range(50):
        now_s = float(i) * dt
        if i > 0:
            pose.x += args.dead_reckon_vx * dt

        points = synthetic_stair_points(
            pose,
            step_x=step_x,
            step_height=step_height,
            visible_x_range=(0.16, 0.95),
            visible_y_range=(-0.50, 0.50),
            resolution=0.05,
        )
        memory.update_from_points(
            points,
            pose,
            now_s,
            min_points_per_cell=args.min_points_per_cell,
            percentile=args.percentile,
            z_min=args.terrain_z_min,
            z_max=args.terrain_z_max,
        )
        memory.prune(pose, now_s)
        support_z = memory.estimate_support_height(
            pose,
            now_s,
            x_range=tuple(args.support_x_range),
            y_range=tuple(args.support_y_range),
            percentile=args.support_percentile,
        )
        if support_z is not None and not args.disable_base_z_estimator:
            apply_base_z_target(
                pose,
                support_z + args.pelvis_height_nominal,
                dt,
                rise_rate=args.base_z_rise_rate,
                fall_rate=args.base_z_fall_rate,
            )
        max_pose_z = max(max_pose_z, pose.z)
        result = memory.render_height_scan(
            pose,
            now_s,
            grid_size=args.grid_size,
            map_size=args.map_size,
            fill_value=args.fill_value,
            offset=args.height_scan_offset,
        )
        final_unknown_ratio = result.unknown_ratio

    expected_min_z = args.pelvis_height_nominal + 0.10
    ok = max_pose_z >= expected_min_z and final_unknown_ratio < 0.55
    print(
        "synthetic_stair: ok=%s final_pose=(%.3f, %.3f, %.3f, %.3f) "
        "max_pose_z=%.3f cells=%d unknown=%.1f%%"
        % (
            ok,
            pose.x,
            pose.y,
            pose.z,
            pose.yaw,
            max_pose_z,
            len(memory.cells),
            final_unknown_ratio * 100.0,
        )
    )
    return 0 if ok else 1


def main(argv=None):
    args, ros_args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test_stair:
        raise SystemExit(run_synthetic_stair_test(args))
    if rclpy is None or point_cloud2 is None:
        raise RuntimeError(
            "ROS2 Python packages are required. Source ROS2 first, for example: "
            "source /opt/ros/humble/setup.bash"
        )
    if args.write_initial_flat:
        initial = np.full(args.grid_size * args.grid_size, args.fill_value, dtype=np.float32)
        write_height_file(args.output, initial)
    rclpy.init(args=ros_args)
    node = HeightMapBridge(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
