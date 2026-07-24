# LiDAR Height Scan Bridge

This policy reads a policy-specific height-scan file:

```bash
/tmp/unitree_g1_lidar_blindwalking_model2600_height_scan.bin
```

Start the bridge before entering `LiDAR_BlindWalking_Model2600`:

```bash
unitree_rl_lab/deploy/robots/g1_29dof/scripts/run_lidar_blindwalking_model2600_heightmap_bridge.sh
```

Defaults:

- subscribes to `/livox/lidar`
- uses `/odom` for x/y/yaw/z motion compensation
- writes an 11x11 `UTHEIGHT1` height scan at the policy-specific path above
- writes viewer/debug metadata to `/tmp/unitree_g1_lidar_blindwalking_model2600_height_scan.csv`

Useful overrides:

```bash
POINTCLOUD_TOPIC=/livox/lidar \
ODOM_TOPIC=/odom \
MOTION_SOURCE=odom \
unitree_rl_lab/deploy/robots/g1_29dof/scripts/run_lidar_blindwalking_model2600_heightmap_bridge.sh
```

For a quick visual check:

```bash
unitree_rl_lab/deploy/robots/g1_29dof/scripts/view_lidar_blindwalking_model2600_height_scan.sh
```

For MuJoCo validation of this same policy, point the simulator height-scan
writer at the same output path. This policy intentionally no longer reads the
shared `/tmp/unitree_g1_height_scan.bin` file.
