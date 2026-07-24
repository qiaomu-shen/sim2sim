#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

HEIGHT_SCAN_BRIDGE_FILE=${HEIGHT_SCAN_BRIDGE_FILE:-/tmp/unitree_g1_lidar_blindwalking_model2600_height_scan.bin}
HEIGHT_SCAN_DEBUG_CSV=${HEIGHT_SCAN_DEBUG_CSV:-/tmp/unitree_g1_lidar_blindwalking_model2600_height_scan.csv}

exec python3 "$SCRIPT_DIR/height_scan_viewer.py" \
  --path "$HEIGHT_SCAN_BRIDGE_FILE" \
  --debug-csv "$HEIGHT_SCAN_DEBUG_CSV" \
  "$@"
