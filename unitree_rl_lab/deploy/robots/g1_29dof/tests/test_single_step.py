from __future__ import annotations

import numpy as np

from heightmap import HeightMapConfig, HeightMapCore, Pose2D, grid_coordinates
from test_flat_ground import PELVIS_Z, points_for_terrain


def test_core_renders_single_step_in_x_forward_direction() -> None:
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

    coords = grid_coordinates()
    terrain = np.where(coords[:, 0] >= 0.0, 0.150, 0.0).astype(np.float32)
    update = core.add_cloud(0.0, points_for_terrain(pose, terrain), now_s=0.0, input_frame="base")

    expected = np.where(coords[:, 0] >= 0.0, 0.143, 0.293).astype(np.float32)
    np.testing.assert_allclose(update.result.values, expected, atol=1.0e-6)
    assert np.count_nonzero(update.result.unknown) == 0
