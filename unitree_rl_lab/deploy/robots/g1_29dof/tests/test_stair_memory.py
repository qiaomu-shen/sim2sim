from __future__ import annotations

import numpy as np

from heightmap import GRID_SIZE, HeightMapConfig, HeightMapCore, Pose2D, synthetic_stair_points

PELVIS_Z = 0.793


def test_front_observation_is_reused_after_step_enters_blind_zone() -> None:
    config = HeightMapConfig(
        input_frame="base",
        fusion_mode="memory",
        voxel=0.0,
        enable_self_filter=False,
        disable_base_z_estimator=True,
        memory_max_age=2.0,
        memory_max_distance=2.5,
        map_query_radius_cells=0,
    )
    core = HeightMapCore(config)

    pose_seen = Pose2D(z=PELVIS_Z)
    core.add_pose(pose_seen)
    points = synthetic_stair_points(
        pose_seen,
        step_x=0.45,
        step_height=0.15,
        visible_x_range=(0.20, 0.95),
        visible_y_range=(-0.50, 0.50),
        resolution=0.05,
    )
    core.add_cloud(0.0, points, now_s=0.0, input_frame="base")

    pose_over_memory = Pose2D(x=0.30, z=PELVIS_Z)
    core.add_pose(pose_over_memory)
    result = core.render_height_scan(now_s=0.7)

    center_y_front_x = 5 * GRID_SIZE + 7
    np.testing.assert_allclose(result.values[center_y_front_x], 0.143, atol=1.0e-6)
    assert result.unknown[center_y_front_x] == 0
    assert result.confidence[center_y_front_x] > 0.0
