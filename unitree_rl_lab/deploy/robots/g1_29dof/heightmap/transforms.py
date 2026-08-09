"""Coordinate transforms and light point-cloud filtering for G1 heightmaps."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Nominal real-robot G1 Mid360 extrinsic from Unitree's official
# unitree_ros G1 rev_1_0 URDF. The fixed Mid360 joint is relative to
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


def copy_pose(pose: Pose2D) -> Pose2D:
    return Pose2D(x=float(pose.x), y=float(pose.y), z=float(pose.z), yaw=float(pose.yaw))


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
    points = np.asarray(points_lidar, dtype=np.float32)
    if points.size == 0:
        return points.reshape(0, 3)
    rot = rotation_from_rpy(*rpy_in_pelvis)
    offset = np.asarray(xyz_in_pelvis, dtype=np.float32)
    return points @ rot.T + offset


def transform_past_base_to_current_base(
    points_base: np.ndarray,
    frame_pose: Pose2D,
    current_pose: Pose2D,
) -> np.ndarray:
    points = np.asarray(points_base, dtype=np.float32)
    if points.size == 0:
        return points.reshape(0, 3)

    cf = math.cos(frame_pose.yaw)
    sf = math.sin(frame_pose.yaw)
    cc = math.cos(current_pose.yaw)
    sc = math.sin(current_pose.yaw)

    xy = points[:, :2]
    world_x = cf * xy[:, 0] - sf * xy[:, 1] + frame_pose.x
    world_y = sf * xy[:, 0] + cf * xy[:, 1] + frame_pose.y

    dx = world_x - current_pose.x
    dy = world_y - current_pose.y

    out = np.empty_like(points)
    out[:, 0] = cc * dx + sc * dy
    out[:, 1] = -sc * dx + cc * dy
    out[:, 2] = points[:, 2] + frame_pose.z - current_pose.z
    return out


def self_filter(points_base: np.ndarray) -> np.ndarray:
    points = np.asarray(points_base, dtype=np.float32)
    if points.size == 0:
        return points.reshape(0, 3)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    torso = (np.abs(x) < 0.34) & (np.abs(y) < 0.30) & (z > -0.18)
    left_leg = (x > -0.26) & (x < 0.22) & (y > 0.035) & (y < 0.22) & (z > -0.82) & (z < -0.10)
    right_leg = (x > -0.26) & (x < 0.22) & (y < -0.035) & (y > -0.22) & (z > -0.82) & (z < -0.10)
    near_feet = (x > -0.12) & (x < 0.18) & (np.abs(y) < 0.22) & (z > -0.88) & (z < -0.64)

    return points[~(torso | left_leg | right_leg | near_feet)]


def voxel_thin(points: np.ndarray, voxel: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if voxel <= 0.0 or len(points) <= 1:
        return points
    keys = np.floor(points / float(voxel)).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]
