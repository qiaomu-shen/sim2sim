#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

POINTCLOUD_TOPIC=${POINTCLOUD_TOPIC:-/livox/lidar}
ODOM_TOPIC=${ODOM_TOPIC:-/odom}
IMU_TOPIC=${IMU_TOPIC:-/imu}
MOTION_SOURCE=${MOTION_SOURCE:-odom}
HEIGHT_SCAN_BRIDGE_FILE=${HEIGHT_SCAN_BRIDGE_FILE:-/tmp/unitree_g1_lidar_blindwalking_model2600_height_scan.bin}
HEIGHT_SCAN_DEBUG_CSV=${HEIGHT_SCAN_DEBUG_CSV:-/tmp/unitree_g1_lidar_blindwalking_model2600_height_scan.csv}

exec python3 "$SCRIPT_DIR/heightmap_bridge.py" \
  --pointcloud-topic "$POINTCLOUD_TOPIC" \
  --odom-topic "$ODOM_TOPIC" \
  --imu-topic "$IMU_TOPIC" \
  --output "$HEIGHT_SCAN_BRIDGE_FILE" \
  --debug-csv "$HEIGHT_SCAN_DEBUG_CSV" \
  --input-frame lidar \
  --fusion-mode memory \
  --motion-source "$MOTION_SOURCE" \
  --grid-size 11 \
  --map-size 1.0 \
  --voxel 0.02 \
  --min-points-per-cell 1 \
  --percentile 90.0 \
  --fill-value 0.28 \
  --write-initial-flat \
  --height-scan-offset 0.5 \
  --terrain-z-min -1.50 \
  --terrain-z-max 0.35 \
  --pelvis-height-nominal 0.793 \
  --memory-resolution 0.025 \
  --memory-max-age 2.0 \
  --memory-max-distance 2.5 \
  --map-query-radius-cells 3 \
  --support-x-range -0.24 0.34 \
  --support-y-range -0.28 0.28 \
  --support-percentile 82.0 \
  --base-z-rise-rate 1.10 \
  --base-z-fall-rate 0.45 \
  --lidar-rpy 3.141592653589793 0.05112069379091391 0.0 \
  "$@"
