from __future__ import annotations

import numpy as np

from heightmap import FILL_VALUE, GRID_SIZE, HEIGHT_SCAN_DIM, encode_height, grid_coordinates

PELVIS_Z = 0.793


def test_flat_ground_encodes_training_value() -> None:
    terrain = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    values = encode_height(PELVIS_Z, terrain)

    assert values.shape == (HEIGHT_SCAN_DIM,)
    np.testing.assert_allclose(values, np.full(HEIGHT_SCAN_DIM, 0.293, dtype=np.float32), atol=1.0e-6)


def test_front_half_step_encodes_lower_height_scan() -> None:
    coords = grid_coordinates()
    terrain = np.where(coords[:, 0] >= 0.0, 0.150, 0.0).astype(np.float32)
    values = encode_height(PELVIS_Z, terrain)

    np.testing.assert_allclose(values[coords[:, 0] >= 0.0], 0.143, atol=1.0e-6)
    np.testing.assert_allclose(values[coords[:, 0] < 0.0], 0.293, atol=1.0e-6)


def test_y_axis_is_not_reversed() -> None:
    coords = grid_coordinates()
    terrain = np.where(coords[:, 1] > 0.0, 0.100, 0.0).astype(np.float32)
    values = encode_height(PELVIS_Z, terrain)

    left_mid = 10 * GRID_SIZE + 5
    right_mid = 0 * GRID_SIZE + 5
    np.testing.assert_allclose(coords[left_mid], [0.0, 0.5], atol=1.0e-6)
    np.testing.assert_allclose(coords[right_mid], [0.0, -0.5], atol=1.0e-6)
    assert values[left_mid] < values[right_mid]
    np.testing.assert_allclose(values[left_mid], 0.193, atol=1.0e-6)
    np.testing.assert_allclose(values[right_mid], 0.293, atol=1.0e-6)


def test_x_axis_is_not_reversed() -> None:
    coords = grid_coordinates()
    terrain = np.where(coords[:, 0] > 0.0, 0.100, 0.0).astype(np.float32)
    values = encode_height(PELVIS_Z, terrain)

    front_mid = 5 * GRID_SIZE + 10
    rear_mid = 5 * GRID_SIZE + 0
    np.testing.assert_allclose(coords[front_mid], [0.5, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(coords[rear_mid], [-0.5, 0.0], atol=1.0e-6)
    assert values[front_mid] < values[rear_mid]
    np.testing.assert_allclose(values[front_mid], 0.193, atol=1.0e-6)
    np.testing.assert_allclose(values[rear_mid], 0.293, atol=1.0e-6)


def test_unknown_cells_emit_fill_value_only_in_final_vector() -> None:
    terrain = np.zeros(HEIGHT_SCAN_DIM, dtype=np.float32)
    terrain[60] = np.nan
    values = encode_height(PELVIS_Z, terrain)

    np.testing.assert_allclose(values[59], 0.293, atol=1.0e-6)
    np.testing.assert_allclose(values[60], FILL_VALUE, atol=1.0e-6)
    np.testing.assert_allclose(values[61], 0.293, atol=1.0e-6)
