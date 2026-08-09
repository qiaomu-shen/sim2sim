"""Fixed 11x11 height_scan contract shared by deployment and tests."""

from __future__ import annotations

import numpy as np

GRID_SIZE = 11
MAP_SIZE = 1.0
HEIGHT_OFFSET = 0.5
FILL_VALUE = 0.28
CLIP_MIN = -1.0
CLIP_MAX = 1.0
HEIGHT_SCAN_DIM = GRID_SIZE * GRID_SIZE


def grid_coordinates(
    *,
    grid_size: int = GRID_SIZE,
    map_size: float = MAP_SIZE,
) -> np.ndarray:
    """Return local pelvis-frame sample coordinates as [N, 2].

    Ordering is exactly the policy input order: y-major, x-fast.
    Coordinates are gravity-level local coordinates, with +x forward and
    +y left. Rotation into world should use yaw only.
    """

    if grid_size < 1:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if map_size <= 0.0:
        raise ValueError(f"map_size must be positive, got {map_size}")

    half = float(map_size) * 0.5
    axis = np.linspace(-half, half, int(grid_size), dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float32)


def encode_height(
    pelvis_world_z: float,
    terrain_world_z: np.ndarray,
    *,
    unknown_mask: np.ndarray | None = None,
    fill_value: float = FILL_VALUE,
    offset: float = HEIGHT_OFFSET,
    clip_min: float = CLIP_MIN,
    clip_max: float = CLIP_MAX,
) -> np.ndarray:
    """Encode terrain heights into the final flattened height_scan vector.

    `terrain_world_z` may be a [121] vector or an [11, 11] y-major grid.
    NaN terrain values, plus any explicit `unknown_mask`, are emitted as
    `fill_value`; unknown is not treated internally as real flat terrain.
    """

    terrain = np.asarray(terrain_world_z, dtype=np.float32)
    terrain_flat = terrain.reshape(-1)

    unknown = ~np.isfinite(terrain_flat)
    if unknown_mask is not None:
        mask = np.asarray(unknown_mask, dtype=bool).reshape(-1)
        if mask.shape != terrain_flat.shape:
            raise ValueError(
                f"unknown_mask shape {mask.shape} does not match terrain shape {terrain_flat.shape}"
            )
        unknown = unknown | mask

    encoded = float(pelvis_world_z) - terrain_flat - float(offset)
    encoded = np.clip(encoded, float(clip_min), float(clip_max)).astype(np.float32)
    if np.any(unknown):
        encoded = encoded.copy()
        encoded[unknown] = np.float32(fill_value)
    return encoded
