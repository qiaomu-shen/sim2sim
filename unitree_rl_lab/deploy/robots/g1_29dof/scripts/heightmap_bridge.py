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
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
except ImportError:
    rclpy = None
    Node = object
    Odometry = object
    PointCloud2 = object
    point_cloud2 = None


MAGIC = "UTHEIGHT1"

# Nominal real-robot G1 mid360 extrinsic from:
#   transfer/obelisk/g1_model/urdf/g1_hand.urdf
#     pelvis -> waist_yaw:      (0, 0, 0)
#     waist_yaw -> waist_roll:  (-0.0039635, 0, 0.035)
#     waist_roll -> torso:      (0, 0, 0.019)
#     torso -> mid360:          (0.0002835, 0.00003, 0.40618), rpy=(0, 0.04014257, 0)
MID360_XYZ_IN_PELVIS = (-0.00368, 0.00003, 0.46018)
MID360_PITCH_RAD = 0.04014257279586953


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


def rotation_y(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def transform_lidar_to_pelvis(
    points_lidar: np.ndarray,
    xyz_in_pelvis: tuple[float, float, float],
    pitch_rad: float,
) -> np.ndarray:
    if points_lidar.size == 0:
        return points_lidar.reshape(0, 3).astype(np.float32, copy=False)
    rot = rotation_y(pitch_rad)
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
) -> tuple[np.ndarray, float]:
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
    return height_obs.reshape(-1), unknown_ratio


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


class HeightMapBridge(Node):
    def __init__(self, args):
        super().__init__("unitree_g1_heightmap_bridge")
        self.args = args
        self.frames: deque[CloudFrame] = deque(maxlen=args.max_frames)
        self.pose = Pose2D(z=args.pelvis_height_nominal)
        self.have_odom = not bool(args.odom_topic)
        self.last_warn_time = 0.0
        self.last_info_time = 0.0

        self.lidar_xyz = tuple(args.lidar_xyz)
        self.lidar_pitch = args.lidar_pitch

        self.cloud_sub = self.create_subscription(
            PointCloud2, args.pointcloud_topic, self.on_cloud, 5
        )
        self.odom_sub = None
        if args.odom_topic:
            self.odom_sub = self.create_subscription(
                Odometry, args.odom_topic, self.on_odom, 20
            )

        self.get_logger().info(
            "bridging %s -> %s, mid360 xyz in pelvis=(%.5f, %.5f, %.5f), pitch=%.5f rad"
            % (
                args.pointcloud_topic,
                args.output,
                self.lidar_xyz[0],
                self.lidar_xyz[1],
                self.lidar_xyz[2],
                self.lidar_pitch,
            )
        )

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

    def on_cloud(self, msg: PointCloud2) -> None:
        now_s = self.get_clock().now().nanoseconds * 1.0e-9
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
            points_base = transform_lidar_to_pelvis(points, self.lidar_xyz, self.lidar_pitch)
        else:
            points_base = points

        points_base = self_filter(points_base)
        self.frames.append(CloudFrame(stamp_s=stamp_s, pose=self.pose, points_base=points_base))

        height_scan, unknown_ratio = fuse_height_scan(
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
        write_height_file(self.args.output, height_scan)

        if now_s - self.last_info_time > self.args.log_interval:
            self.get_logger().info(
                "height_scan min=%.3f max=%.3f unknown=%.1f%% frames=%d points=%d"
                % (
                    float(np.min(height_scan)),
                    float(np.max(height_scan)),
                    unknown_ratio * 100.0,
                    len(self.frames),
                    len(points_base),
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
            "height_scan bridge file expected by the 21-DOF policy."
        )
    )
    parser.add_argument("--pointcloud-topic", default="/livox/lidar")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--output", default="/tmp/unitree_g1_height_scan.bin")
    parser.add_argument("--input-frame", choices=("lidar", "base"), default="lidar")
    parser.add_argument("--grid-size", type=int, default=11)
    parser.add_argument("--map-size", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-age", type=float, default=0.8)
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--min-points-per-cell", type=int, default=1)
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--fill-value", type=float, default=0.35)
    parser.add_argument("--pelvis-height-nominal", type=float, default=0.793)
    parser.add_argument(
        "--lidar-xyz",
        type=float,
        nargs=3,
        default=list(MID360_XYZ_IN_PELVIS),
        metavar=("X", "Y", "Z"),
        help="mid360 origin in pelvis frame, meters.",
    )
    parser.add_argument(
        "--lidar-pitch",
        type=float,
        default=MID360_PITCH_RAD,
        help="mid360 pitch relative to pelvis/torso, radians.",
    )
    parser.add_argument("--log-interval", type=float, default=1.0)
    return parser.parse_known_args(argv)


def main(argv=None):
    args, ros_args = parse_args(sys.argv[1:] if argv is None else argv)
    if rclpy is None or point_cloud2 is None:
        raise RuntimeError(
            "ROS2 Python packages are required. Source ROS2 first, for example: "
            "source /opt/ros/humble/setup.bash"
        )
    rclpy.init(args=ros_args)
    node = HeightMapBridge(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
