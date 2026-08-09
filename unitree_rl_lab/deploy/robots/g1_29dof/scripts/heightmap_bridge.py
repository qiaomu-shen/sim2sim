#!/usr/bin/env python3

import argparse
import math
import os
import sys
import time
from typing import Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROBOT_DIR not in sys.path:
    sys.path.insert(0, ROBOT_DIR)

from heightmap import (  # noqa: E402
    MID360_RPY_IN_PELVIS,
    MID360_XYZ_IN_PELVIS,
    HeightMapConfig,
    HeightMapCore,
    Pose2D,
    synthetic_stair_points,
    yaw_from_quat_xyzw,
)

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
    confidence: Optional[np.ndarray] = None,
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
    confidence_mean = "nan"
    if confidence is not None and len(confidence) == len(values):
        confidence_mean = f"{float(np.mean(confidence)):.6f}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(f"mode,{mode}\n")
        f.write(f"est_pose,{pose.x:.6f} {pose.y:.6f} {pose.z:.6f} {pose.yaw:.6f}\n")
        f.write(f"support_z,{support_text}\n")
        f.write(f"unknown_ratio,{unknown_ratio:.6f}\n")
        f.write(f"confidence_mean,{confidence_mean}\n")
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
        if confidence is not None and len(confidence) == len(values):
            f.write("confidence\n")
            for iy in range(grid_size):
                row = confidence[iy * grid_size : (iy + 1) * grid_size]
                f.write(",".join(f"{float(v):.6f}" for v in row))
                f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def core_config_from_args(args, lidar_xyz, lidar_rpy) -> HeightMapConfig:
    return HeightMapConfig(
        input_frame=args.input_frame,
        fusion_mode=args.fusion_mode,
        grid_size=args.grid_size,
        map_size=args.map_size,
        max_frames=args.max_frames,
        max_age=args.max_age,
        voxel=args.voxel,
        min_points_per_cell=args.min_points_per_cell,
        percentile=args.percentile,
        fill_value=args.fill_value,
        height_scan_offset=args.height_scan_offset,
        terrain_z_min=args.terrain_z_min,
        terrain_z_max=args.terrain_z_max,
        pelvis_height_nominal=args.pelvis_height_nominal,
        lidar_xyz=lidar_xyz,
        lidar_rpy=lidar_rpy,
        memory_resolution=args.memory_resolution,
        memory_max_age=args.memory_max_age,
        memory_max_distance=args.memory_max_distance,
        map_query_radius_cells=args.map_query_radius_cells,
        disable_base_z_estimator=args.disable_base_z_estimator,
        support_x_range=tuple(args.support_x_range),
        support_y_range=tuple(args.support_y_range),
        support_percentile=args.support_percentile,
        base_z_rise_rate=args.base_z_rise_rate,
        base_z_fall_rate=args.base_z_fall_rate,
    )


class HeightMapBridge(Node):
    def __init__(self, args):
        super().__init__("unitree_g1_heightmap_bridge")
        self.args = args
        self.have_odom = args.motion_source != "odom" or not bool(args.odom_topic)
        self.last_warn_time = 0.0
        self.last_info_time = 0.0
        self.last_motion_time: Optional[float] = None
        self.imu_yaw_rate = 0.0

        self.lidar_xyz = tuple(args.lidar_xyz)
        self.lidar_rpy = tuple(args.lidar_rpy)
        if args.lidar_pitch is not None:
            self.lidar_rpy = (0.0, args.lidar_pitch, 0.0)
        self.core = HeightMapCore(core_config_from_args(args, self.lidar_xyz, self.lidar_rpy))
        self.pose = self.core.pose

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
        self.core.add_pose(self.pose)

    def on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        self.pose = Pose2D(
            x=float(p.x),
            y=float(p.y),
            z=float(p.z),
            yaw=yaw_from_quat_xyzw(float(q.x), float(q.y), float(q.z), float(q.w)),
        )
        self.core.add_pose(self.pose)
        self.pose = self.core.pose
        self.have_odom = True

    def on_imu(self, msg: Imu) -> None:
        self.imu_yaw_rate = float(msg.angular_velocity.z)

    def on_cloud(self, msg: PointCloud2) -> None:
        now_s = self.get_clock().now().nanoseconds * 1.0e-9
        self.update_dead_reckon(now_s)
        if not self.have_odom and self.core.frames:
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

        update = self.core.add_cloud(stamp_s, points, now_s=now_s)
        self.pose = update.pose
        result = update.result
        height_scan = result.values
        unknown_ratio = result.unknown_ratio
        write_height_file(self.args.output, height_scan)
        write_debug_csv(
            self.args.debug_csv,
            height_scan,
            result.unknown,
            confidence=result.confidence,
            mode=self.args.fusion_mode,
            pose=update.pose,
            unknown_ratio=unknown_ratio,
            memory_cells=update.memory_cells,
            map_updates=update.map_updates,
            point_count=update.point_count,
            support_z=update.support_z,
        )

        if now_s - self.last_info_time > self.args.log_interval:
            self.get_logger().info(
                "height_scan min=%.3f max=%.3f unknown=%.1f%% frames=%d points=%d map_cells=%d map_updates=%d pose=(%.3f, %.3f, %.3f, %.3f) support_z=%s"
                % (
                    float(np.min(height_scan)),
                    float(np.max(height_scan)),
                    unknown_ratio * 100.0,
                    update.frame_count,
                    update.point_count,
                    update.memory_cells,
                    update.map_updates,
                    update.pose.x,
                    update.pose.y,
                    update.pose.z,
                    update.pose.yaw,
                    "nan" if update.support_z is None else f"{update.support_z:.3f}",
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
    parser.add_argument("--memory-resolution", type=float, default=0.025)
    parser.add_argument("--memory-max-age", type=float, default=2.0)
    parser.add_argument("--memory-max-distance", type=float, default=2.5)
    parser.add_argument("--map-query-radius-cells", type=int, default=3)
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


def run_synthetic_stair_test(args) -> int:
    lidar_xyz = tuple(args.lidar_xyz)
    lidar_rpy = (0.0, args.lidar_pitch, 0.0) if args.lidar_pitch is not None else tuple(args.lidar_rpy)
    config = core_config_from_args(args, lidar_xyz, lidar_rpy)
    config.fusion_mode = "memory"
    config.input_frame = "base"
    core = HeightMapCore(config)
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

        core.add_pose(pose)
        points = synthetic_stair_points(
            pose,
            step_x=step_x,
            step_height=step_height,
            visible_x_range=(0.16, 0.95),
            visible_y_range=(-0.50, 0.50),
            resolution=0.05,
        )
        update = core.add_cloud(now_s, points, now_s=now_s, input_frame="base")
        pose = update.pose
        max_pose_z = max(max_pose_z, update.pose.z)
        final_unknown_ratio = update.result.unknown_ratio

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
            len(core.terrain_memory.cells),
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
