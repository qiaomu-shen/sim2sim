# G1 29-DOF Height Scan Contract

This document fixes the deployment-side contract for the policy's 121-dimensional
`height_scan`. Do not tune the rolling map or fill value until these rules pass
the unit tests against training-side expectations.

## Grid

- `GRID_SIZE = 11`
- `MAP_SIZE = 1.0 m`
- The grid is centered on the horizontal projection of the current pelvis.
- Local sample coordinates span `x, y in [-0.5, 0.5] m`.
- The spacing is `1.0 / (11 - 1) = 0.1 m`.
- `+x` is robot forward, `+y` is robot left, `+z` is up.
- The query grid rotates with pelvis yaw only. It must not tilt with body roll or pitch.

## Flattening

The vector is y-major and x-fast:

```python
for iy in range(11):
    for ix in range(11):
        output.append(grid[iy, ix])
```

Thus element `0` is `(-0.5, -0.5)`, element `60` is `(0.0, 0.0)`,
and element `120` is `(0.5, 0.5)`.

## Height Encoding

Each known terrain cell is encoded as:

```text
height_scan = pelvis_world_z - terrain_world_z - 0.5
```

Then the value is clipped to `[-1.0, 1.0]`.

With `pelvis_world_z = 0.793 m`:

- Flat ground at `terrain_world_z = 0.000 m` encodes to `0.293`.
- A `0.150 m` step encodes to `0.143`.

Unknown cells are emitted as `0.28` for the policy input, but the algorithm also
keeps an `unknown_mask` and `confidence` vector. Unknown cells must not be
treated internally as measured flat terrain.
