from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def read_rows(path: Path) -> list[dict[str, str]]:
  if not path.exists():
    return []
  with path.open(encoding="utf-8", newline="") as f:
    return list(csv.DictReader(f))


def latest_run_dir(log_root: Path) -> Path:
  latest_file = log_root / "latest_run_dir.txt"
  if not latest_file.exists():
    return log_root
  try:
    run_dir = Path(latest_file.read_text(encoding="utf-8").strip())
  except OSError:
    return log_root
  if run_dir.exists() and run_dir.is_dir():
    return run_dir
  return log_root


def to_float(row: dict[str, str], name: str, default: float = 0.0) -> float:
  try:
    return float(row.get(name, default))
  except (TypeError, ValueError):
    return default


def to_int(row: dict[str, str], name: str, default: int = 0) -> int:
  try:
    return int(float(row.get(name, default)))
  except (TypeError, ValueError):
    return default


def percentile(values: list[float], q: float) -> float | None:
  values = sorted(values)
  if not values:
    return None
  if len(values) == 1:
    return values[0]
  pos = (len(values) - 1) * q
  lo = int(pos)
  hi = min(lo + 1, len(values) - 1)
  frac = pos - lo
  return values[lo] * (1.0 - frac) + values[hi] * frac


def event_tick(row: dict[str, str]) -> int:
  if "event_tick_ms" in row:
    return to_int(row, "event_tick_ms")
  return to_int(row, "tick_ms")


def event_time(row: dict[str, str]) -> float:
  if "event_time_s" in row:
    return to_float(row, "event_time_s")
  return to_float(row, "time_s")


def frame_tick_window(path: Path) -> tuple[int | None, int | None]:
  ticks = [
    to_int(row, "lowstate_tick_ms", -1)
    for row in read_rows(path)
    if to_int(row, "lowstate_tick_ms", -1) >= 0
  ]
  if not ticks:
    return None, None
  return min(ticks), max(ticks)


def command_norm(row: dict[str, str]) -> float:
  return max(
    abs(to_float(row, "cmd_x")),
    abs(to_float(row, "cmd_y")),
    abs(to_float(row, "cmd_yaw")),
  )


def moving_tick_windows(
  path: Path,
  min_command_norm: float,
  merge_gap_ms: int,
  pad_ms: int,
) -> list[tuple[int, int]]:
  moving_ticks = [
    to_int(row, "lowstate_tick_ms", -1)
    for row in read_rows(path)
    if to_int(row, "lowstate_tick_ms", -1) >= 0
    and command_norm(row) >= min_command_norm
  ]
  if not moving_ticks:
    return []

  windows: list[tuple[int, int]] = []
  start = moving_ticks[0]
  prev = moving_ticks[0]
  for tick in moving_ticks[1:]:
    if tick - prev <= merge_gap_ms:
      prev = tick
      continue
    windows.append((max(0, start - pad_ms), prev + pad_ms))
    start = tick
    prev = tick
  windows.append((max(0, start - pad_ms), prev + pad_ms))
  return windows


def filter_by_tick_window(
  rows: list[dict[str, str]],
  start_tick_ms: int | None,
  end_tick_ms: int | None,
) -> list[dict[str, str]]:
  filtered = []
  for row in rows:
    tick = event_tick(row)
    if start_tick_ms is not None and tick < start_tick_ms:
      continue
    if end_tick_ms is not None and tick > end_tick_ms:
      continue
    filtered.append(row)
  return filtered


def filter_by_tick_windows(
  rows: list[dict[str, str]],
  windows: list[tuple[int, int]],
) -> list[dict[str, str]]:
  if not windows:
    return []
  filtered = []
  for row in rows:
    tick = event_tick(row)
    if any(start <= tick <= end for start, end in windows):
      filtered.append(row)
  return filtered


def match_events(
  truth_rows: list[dict[str, str]],
  detector_rows: list[dict[str, str]],
  tolerance_ms: int,
) -> tuple[list[dict[str, object]], list[dict[str, str]], list[dict[str, str]]]:
  matches: list[dict[str, object]] = []
  unmatched_detector = set(range(len(detector_rows)))

  for truth in sorted(truth_rows, key=event_tick):
    truth_tick = event_tick(truth)
    foot = truth.get("foot", "")
    best_idx: int | None = None
    best_abs_error: int | None = None

    for idx in sorted(unmatched_detector):
      det = detector_rows[idx]
      if det.get("foot", "") != foot:
        continue
      error = event_tick(det) - truth_tick
      abs_error = abs(error)
      if abs_error > tolerance_ms:
        continue
      if best_abs_error is None or abs_error < best_abs_error:
        best_idx = idx
        best_abs_error = abs_error

    if best_idx is None:
      continue

    detector = detector_rows[best_idx]
    unmatched_detector.remove(best_idx)
    signed_error_ms = event_tick(detector) - truth_tick
    matches.append(
      {
        "foot": foot,
        "truth_tick_ms": truth_tick,
        "detector_tick_ms": event_tick(detector),
        "signed_error_ms": signed_error_ms,
        "abs_error_ms": abs(signed_error_ms),
        "truth_time_s": event_time(truth),
        "detector_time_s": event_time(detector),
        "detector_score": to_int(detector, "score"),
        "detector_confidence": detector.get("confidence", ""),
        "truth_normal_force": to_float(truth, "normal_force"),
        "truth_contact_count": to_int(truth, "contact_count"),
      }
    )

  matched_truth_keys = {(match["foot"], match["truth_tick_ms"]) for match in matches}
  missed_truth = [
    row
    for row in truth_rows
    if (row.get("foot", ""), event_tick(row)) not in matched_truth_keys
  ]
  false_detector = [detector_rows[idx] for idx in sorted(unmatched_detector)]
  return matches, missed_truth, false_detector


def summarize(
  truth_rows: list[dict[str, str]],
  detector_rows: list[dict[str, str]],
  matches: list[dict[str, object]],
  missed_truth: list[dict[str, str]],
  false_detector: list[dict[str, str]],
  tolerance_ms: int,
) -> dict[str, object]:
  errors = [float(match["signed_error_ms"]) for match in matches]
  abs_errors = [float(match["abs_error_ms"]) for match in matches]
  truth_count = len(truth_rows)
  detector_count = len(detector_rows)
  matched = len(matches)
  return {
    "tolerance_ms": tolerance_ms,
    "truth_events": truth_count,
    "detector_events": detector_count,
    "matched_events": matched,
    "missed_truth_events": len(missed_truth),
    "false_detector_events": len(false_detector),
    "precision": matched / detector_count if detector_count else None,
    "recall": matched / truth_count if truth_count else None,
    "signed_error_ms_mean": mean(errors) if errors else None,
    "signed_error_ms_median": median(errors) if errors else None,
    "abs_error_ms_mean": mean(abs_errors) if abs_errors else None,
    "abs_error_ms_median": median(abs_errors) if abs_errors else None,
    "abs_error_ms_p90": percentile(abs_errors, 0.90),
    "truth_by_foot": {
      foot: sum(1 for row in truth_rows if row.get("foot") == foot)
      for foot in ("left", "right")
    },
    "detector_by_foot": {
      foot: sum(1 for row in detector_rows if row.get("foot") == foot)
      for foot in ("left", "right")
    },
    "matched_by_foot": {
      foot: sum(1 for row in matches if row.get("foot") == foot)
      for foot in ("left", "right")
    },
  }


def write_matches(path: Path, rows: list[dict[str, object]]) -> None:
  fieldnames = [
    "foot",
    "truth_tick_ms",
    "detector_tick_ms",
    "signed_error_ms",
    "abs_error_ms",
    "truth_time_s",
    "detector_time_s",
    "detector_score",
    "detector_confidence",
    "truth_normal_force",
    "truth_contact_count",
  ]
  with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  g1_dir = Path(__file__).resolve().parents[1]
  default_log_dir = latest_run_dir(g1_dir / "log")
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-dir", type=Path, default=default_log_dir)
  parser.add_argument("--truth", type=Path, default=None)
  parser.add_argument("--detector", type=Path, default=None)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument(
    "--frame-log",
    type=Path,
    default=None,
  )
  parser.add_argument("--tolerance-ms", type=int, default=80)
  parser.add_argument("--eval-start-tick-ms", type=int, default=None)
  parser.add_argument("--eval-end-tick-ms", type=int, default=None)
  parser.add_argument("--no-frame-window", action="store_true")
  parser.add_argument("--include-stationary", action="store_true")
  parser.add_argument("--min-command-norm", type=float, default=0.05)
  parser.add_argument("--moving-merge-gap-ms", type=int, default=100)
  parser.add_argument("--moving-pad-ms", type=int, default=0)
  args = parser.parse_args()

  truth_path = args.truth or args.log_dir / "touchdown_truth.csv"
  detector_path = args.detector or args.log_dir / "touchdown_events.csv"
  frame_log_path = args.frame_log or args.log_dir / "mjlab_g1_touchdown_log.csv"
  out_dir = args.out_dir or args.log_dir

  raw_truth_rows = [
    row for row in read_rows(truth_path) if row.get("event_type") == "sim_touchdown"
  ]
  raw_detector_rows = [
    row
    for row in read_rows(detector_path)
    if row.get("event_type") == "detector_touchdown"
  ]

  eval_start_tick_ms = args.eval_start_tick_ms
  eval_end_tick_ms = args.eval_end_tick_ms
  if not args.no_frame_window:
    frame_start, frame_end = frame_tick_window(frame_log_path)
    if eval_start_tick_ms is None:
      eval_start_tick_ms = frame_start
    if eval_end_tick_ms is None:
      eval_end_tick_ms = frame_end

  truth_rows = filter_by_tick_window(
    raw_truth_rows,
    eval_start_tick_ms,
    eval_end_tick_ms,
  )
  detector_rows = filter_by_tick_window(
    raw_detector_rows,
    eval_start_tick_ms,
    eval_end_tick_ms,
  )
  moving_windows: list[tuple[int, int]] = []
  if not args.include_stationary:
    moving_windows = moving_tick_windows(
      frame_log_path,
      args.min_command_norm,
      args.moving_merge_gap_ms,
      args.moving_pad_ms,
    )
    truth_rows = filter_by_tick_windows(truth_rows, moving_windows)
    detector_rows = filter_by_tick_windows(detector_rows, moving_windows)

  out_dir.mkdir(parents=True, exist_ok=True)
  matches, missed_truth, false_detector = match_events(
    truth_rows,
    detector_rows,
    args.tolerance_ms,
  )
  summary = summarize(
    truth_rows,
    detector_rows,
    matches,
    missed_truth,
    false_detector,
    args.tolerance_ms,
  )
  summary.update(
    {
      "raw_truth_events": len(raw_truth_rows),
      "raw_detector_events": len(raw_detector_rows),
      "eval_start_tick_ms": eval_start_tick_ms,
      "eval_end_tick_ms": eval_end_tick_ms,
      "frame_log": str(frame_log_path),
      "moving_filter_enabled": not args.include_stationary,
      "min_command_norm": args.min_command_norm,
      "moving_merge_gap_ms": args.moving_merge_gap_ms,
      "moving_pad_ms": args.moving_pad_ms,
      "moving_window_count": len(moving_windows),
      "moving_window_start_tick_ms": moving_windows[0][0] if moving_windows else None,
      "moving_window_end_tick_ms": moving_windows[-1][1] if moving_windows else None,
    }
  )

  matches_path = out_dir / "touchdown_accuracy_matches.csv"
  summary_json_path = out_dir / "touchdown_accuracy_summary.json"
  summary_txt_path = out_dir / "touchdown_accuracy_summary.txt"
  write_matches(matches_path, matches)
  with summary_json_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
    f.write("\n")

  lines = [
    "Touchdown detector accuracy",
    f"log_dir: {args.log_dir}",
    f"truth_csv: {truth_path}",
    f"detector_csv: {detector_path}",
    f"frame_log: {frame_log_path}",
    f"tolerance_ms: {args.tolerance_ms}",
    (
      "eval_window_ms: "
      f"start={summary['eval_start_tick_ms']} "
      f"end={summary['eval_end_tick_ms']}"
    ),
    (
      "raw_events: "
      f"truth={summary['raw_truth_events']} "
      f"detector={summary['raw_detector_events']}"
    ),
    (
      "moving_filter: "
      f"enabled={summary['moving_filter_enabled']} "
      f"min_command_norm={summary['min_command_norm']} "
      f"windows={summary['moving_window_count']} "
      f"start={summary['moving_window_start_tick_ms']} "
      f"end={summary['moving_window_end_tick_ms']}"
    ),
    (
      "events: "
      f"truth={summary['truth_events']} "
      f"detector={summary['detector_events']} "
      f"matched={summary['matched_events']} "
      f"missed={summary['missed_truth_events']} "
      f"false_positive={summary['false_detector_events']}"
    ),
    (f"metrics: precision={summary['precision']} recall={summary['recall']}"),
    (
      "timing_ms: "
      f"signed_median={summary['signed_error_ms_median']} "
      f"abs_median={summary['abs_error_ms_median']} "
      f"abs_p90={summary['abs_error_ms_p90']}"
    ),
    f"truth_by_foot: {summary['truth_by_foot']}",
    f"detector_by_foot: {summary['detector_by_foot']}",
    f"matched_by_foot: {summary['matched_by_foot']}",
    f"matches_csv: {matches_path}",
    f"summary_json: {summary_json_path}",
  ]
  with summary_txt_path.open("w", encoding="utf-8") as f:
    f.write("\n".join(lines))
    f.write("\n")
  print("\n".join(lines))


if __name__ == "__main__":
  main()
