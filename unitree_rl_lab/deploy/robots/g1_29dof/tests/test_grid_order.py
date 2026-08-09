from __future__ import annotations

import numpy as np

from heightmap import GRID_SIZE, grid_coordinates


def test_first_last_and_center_coordinates_are_fixed() -> None:
    coords = grid_coordinates()

    assert coords.shape == (GRID_SIZE * GRID_SIZE, 2)
    np.testing.assert_allclose(coords[0], [-0.5, -0.5], atol=1.0e-6)
    np.testing.assert_allclose(coords[60], [0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(coords[-1], [0.5, 0.5], atol=1.0e-6)


def test_flattening_is_y_major_x_fast() -> None:
    coords = grid_coordinates()

    np.testing.assert_allclose(coords[1] - coords[0], [0.1, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(coords[GRID_SIZE] - coords[0], [0.0, 0.1], atol=1.0e-6)
    for iy in range(GRID_SIZE):
        row = coords[iy * GRID_SIZE : (iy + 1) * GRID_SIZE]
        np.testing.assert_allclose(row[:, 1], -0.5 + 0.1 * iy, atol=1.0e-6)
        np.testing.assert_allclose(row[:, 0], np.linspace(-0.5, 0.5, GRID_SIZE), atol=1.0e-6)
