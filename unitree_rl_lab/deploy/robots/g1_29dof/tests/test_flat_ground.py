from __future__ import annotations

import numpy as np

from heightmap import GRID_SIZE, HeightMapConfig, HeightMapCore, Pose2D, grid_coordinates

PELVIS_Z = 0.793


def points_for_terrain(pose: Pose2D, terrain_world_z: np.ndarray) -> np.ndarray:
    coords = grid_coordinates()
    return np.column_stack((coords[:, 0], coords[:, 1], terrain_world_z - pose.z)).astype(np.float32)


def test_core_renders_flat_ground_without_ros() -> None:
    config = HeightMapConfig(
        input_frame="base",
        fusion_mode="memory",
        voxel=0.0,
        enable_self_filter=False,
        disable_base_z_estimator=True,
        map_query_radius_cells=0,
    )
    core = HeightMapCore(config)
    pose = Pose2D(z=PELVIS_Z)
    core.add_pose(pose)

    terrain = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
    update = core.add_cloud(0.0, points_for_terrain(pose, terrain), now_s=0.0, input_frame="base")

    np.testing.assert_allclose(update.result.values, 0.293, atol=1.0e-6)
    assert np.count_nonzero(update.result.unknown) == 0
    np.testing.assert_allclose(update.result.confidence, 1.0, atol=1.0e-6)
    assert update.result.unknown_ratio == 0.0
    assert update.memory_cells == GRID_SIZE * GRID_SIZE
