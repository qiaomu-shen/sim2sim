# Branch Notes

This file records the intended role of each branch when the workspace is pushed
to GitHub.

| Branch | Purpose |
| --- | --- |
| `main` | Repository entry point, shared layout notes, and branch index. |
| `test/lidar-blindzone-heightmap` | G1-29DOF lidar blindwalking deployment work with heightmap bridge, real/sim diagnostics, and 21DOF policy migration notes. |

## Current Notes

- Keep branch-specific run commands in that branch's `README.md`.
- Keep robot-specific safety checks close to deployment scripts and policy
  config files.
- Do not merge experimental robot-control changes back to `main` until the
  real/sim observation order and joint mapping have been verified.
