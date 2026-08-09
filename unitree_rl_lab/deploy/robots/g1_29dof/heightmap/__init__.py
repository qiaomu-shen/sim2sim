"""ROS-independent heightmap components for the G1 29-DOF deploy bridge."""

from .height_scan_contract import (
    CLIP_MAX,
    CLIP_MIN,
    FILL_VALUE,
    GRID_SIZE,
    HEIGHT_OFFSET,
    HEIGHT_SCAN_DIM,
    MAP_SIZE,
    encode_height,
    grid_coordinates,
)
from .heightmap_core import HeightMapConfig, HeightMapCore, HeightMapUpdate, synthetic_stair_points
from .terrain_memory import HeightScanResult, TerrainCell, TerrainMemory, TerrainQuery, TerrainSource
from .transforms import (
    MID360_RPY_IN_PELVIS,
    MID360_XYZ_IN_PELVIS,
    Pose2D,
    rotation_from_rpy,
    self_filter,
    transform_lidar_to_pelvis,
    transform_past_base_to_current_base,
    voxel_thin,
    yaw_from_quat_xyzw,
)

__all__ = [
    "CLIP_MAX",
    "CLIP_MIN",
    "FILL_VALUE",
    "GRID_SIZE",
    "HEIGHT_OFFSET",
    "HEIGHT_SCAN_DIM",
    "MAP_SIZE",
    "MID360_RPY_IN_PELVIS",
    "MID360_XYZ_IN_PELVIS",
    "HeightMapConfig",
    "HeightMapCore",
    "HeightMapUpdate",
    "HeightScanResult",
    "Pose2D",
    "TerrainCell",
    "TerrainMemory",
    "TerrainQuery",
    "TerrainSource",
    "encode_height",
    "grid_coordinates",
    "rotation_from_rpy",
    "self_filter",
    "synthetic_stair_points",
    "transform_lidar_to_pelvis",
    "transform_past_base_to_current_base",
    "voxel_thin",
    "yaw_from_quat_xyzw",
]
