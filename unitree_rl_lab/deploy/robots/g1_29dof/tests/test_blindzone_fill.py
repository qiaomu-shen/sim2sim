from __future__ import annotations

import numpy as np

from heightmap import FILL_VALUE, HEIGHT_SCAN_DIM, HeightMapConfig, HeightMapCore, Pose2D


def test_unknown_cells_keep_mask_and_confidence_with_policy_fill_value() -> None:
    config = HeightMapConfig(
        input_frame="base",
        fusion_mode="memory",
        enable_self_filter=False,
        disable_base_z_estimator=True,
    )
    core = HeightMapCore(config)
    core.add_pose(Pose2D(z=0.793))

    result = core.render_height_scan(now_s=0.0)

    np.testing.assert_allclose(result.values, np.full(HEIGHT_SCAN_DIM, FILL_VALUE), atol=1.0e-6)
    np.testing.assert_array_equal(result.unknown, np.ones(HEIGHT_SCAN_DIM, dtype=np.uint8))
    np.testing.assert_allclose(result.confidence, 0.0, atol=1.0e-6)
    assert result.unknown_ratio == 1.0
