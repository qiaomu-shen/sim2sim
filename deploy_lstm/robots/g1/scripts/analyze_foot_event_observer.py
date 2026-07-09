from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import analyze_touchdown_accuracy as base


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


def observer_probability(row: dict[str, str]) -> float:
  return base.to_float(row, "probability")


def infer_contact_tick_offset(
  contact_rows: list[dict[str, str]],
  frame_log_path: Path,
) -> int:
  """Infer whether contact_state tick_ms is local sim time or lowstate time."""
  if not contact_rows:
    return 0
  frame_start, _frame_end = base.frame_tick_window(frame_log_path)
  if frame_start is None:
    return 0
  first_contact_tick = min(base.event_tick(row) for row in contact_rows)
  if abs(first_contact_tick - frame_start) <= 1000:
    return 0
  return frame_start - first_contact_tick


class ContactStateIndex:
  """Nearest-neighbor lookup for sim contact_state rows."""

  def __init__(
    self,
    rows: list[dict[str, str]],
    *,
    tick_offset_ms: int,
  ) -> None:
    self._rows: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    self._ticks: dict[str, list[int]] = {"left": [], "right": []}
    for row in rows:
      foot = row.get("foot", "")
      if foot not in self._rows:
        continue
      tick = base.event_tick(row) + tick_offset_ms
      enriched = dict(row)
      enriched["aligned_tick_ms"] = tick
      self._rows[foot].append(enriched)
    for foot, foot_rows in self._rows.items():
      foot_rows.sort(key=lambda row: int(row["aligned_tick_ms"]))
      self._ticks[foot] = [int(row["aligned_tick_ms"]) for row in foot_rows]

  def nearest_state(
    self,
    foot: str,
    tick_ms: int,
  ) -> tuple[dict[str, Any] | None, int | None]:
    rows = self._rows.get(foot, [])
    ticks = self._ticks.get(foot, [])
    if not rows or not ticks:
      return None, None
    pos = bisect.bisect_left(ticks, tick_ms)
    candidates = []
    if pos < len(rows):
      candidates.append(pos)
    if pos > 0:
      candidates.append(pos - 1)
    best_pos = min(candidates, key=lambda idx: abs(ticks[idx] - tick_ms))
    return rows[best_pos], ticks[best_pos] - tick_ms

  def contact_window(
    self,
    foot: str,
    tick_ms: int,
    radius_ms: int,
  ) -> list[dict[str, Any]]:
    rows = self._rows.get(foot, [])
    ticks = self._ticks.get(foot, [])
    if not rows or not ticks:
      return []
    start = tick_ms - radius_ms
    end = tick_ms + radius_ms
    lo = bisect.bisect_left(ticks, start)
    hi = bisect.bisect_right(ticks, end)
    return rows[lo:hi]

  def nearest_contact(
    self,
    foot: str,
    tick_ms: int,
  ) -> tuple[dict[str, Any] | None, int | None]:
    contact_rows = [
      row
      for row in self._rows.get(foot, [])
      if base.to_int(row, "contact_count") > 0
    ]
    if not contact_rows:
      return None, None
    best = min(
      contact_rows,
      key=lambda row: abs(int(row["aligned_tick_ms"]) - tick_ms),
    )
    return best, int(best["aligned_tick_ms"]) - tick_ms


def nearest_truth_diagnostics(
  truth_rows: list[dict[str, str]],
  foot: str,
  tick_ms: int,
) -> dict[str, Any]:
  same_foot = sorted(
    [row for row in truth_rows if row.get("foot", "") == foot],
    key=base.event_tick,
  )
  if not same_foot:
    return {
      "prev_truth_touchdown_dt_ms": "",
      "next_truth_touchdown_dt_ms": "",
      "nearest_truth_touchdown_dt_ms": "",
      "nearest_truth_touchdown_abs_dt_ms": "",
      "nearest_truth_normal_force": "",
      "nearest_truth_contact_count": "",
      "nearest_truth_other_geom_name": "",
    }
  prev_rows = [row for row in same_foot if base.event_tick(row) < tick_ms]
  next_rows = [row for row in same_foot if base.event_tick(row) > tick_ms]
  nearest = min(same_foot, key=lambda row: abs(base.event_tick(row) - tick_ms))
  nearest_dt = base.event_tick(nearest) - tick_ms
  return {
    "prev_truth_touchdown_dt_ms": (
      tick_ms - base.event_tick(prev_rows[-1]) if prev_rows else ""
    ),
    "next_truth_touchdown_dt_ms": (
      base.event_tick(next_rows[0]) - tick_ms if next_rows else ""
    ),
    "nearest_truth_touchdown_dt_ms": nearest_dt,
    "nearest_truth_touchdown_abs_dt_ms": abs(nearest_dt),
    "nearest_truth_normal_force": base.to_float(nearest, "normal_force"),
    "nearest_truth_contact_count": base.to_int(nearest, "contact_count"),
    "nearest_truth_other_geom_name": nearest.get("other_geom_name", ""),
  }


def contact_diagnostics(
  contact_index: ContactStateIndex,
  foot: str,
  tick_ms: int,
  radius_ms: int,
) -> dict[str, Any]:
  nearest_state, nearest_state_dt = contact_index.nearest_state(foot, tick_ms)
  nearest_contact, nearest_contact_dt = contact_index.nearest_contact(foot, tick_ms)
  window = contact_index.contact_window(foot, tick_ms, radius_ms)
  contact_window = [
    row for row in window if base.to_int(row, "contact_count") > 0
  ]
  return {
    "nearest_contact_state_dt_ms": nearest_state_dt if nearest_state else "",
    "contact_count_at_nearest_state": (
      base.to_int(nearest_state, "contact_count") if nearest_state else ""
    ),
    "normal_force_at_nearest_state": (
      base.to_float(nearest_state, "normal_force") if nearest_state else ""
    ),
    "other_geom_at_nearest_state": (
      nearest_state.get("other_geom_name", "") if nearest_state else ""
    ),
    "contact_frames_within_window": len(contact_window),
    "contact_any_within_window": int(bool(contact_window)),
    "normal_force_max_within_window": max(
      (base.to_float(row, "normal_force") for row in contact_window),
      default=0.0,
    ),
    "nearest_contact_dt_ms": nearest_contact_dt if nearest_contact else "",
    "nearest_contact_abs_dt_ms": (
      abs(nearest_contact_dt) if nearest_contact_dt is not None else ""
    ),
    "nearest_contact_normal_force": (
      base.to_float(nearest_contact, "normal_force") if nearest_contact else ""
    ),
    "nearest_contact_other_geom_name": (
      nearest_contact.get("other_geom_name", "") if nearest_contact else ""
    ),
  }


def enrich_false_observer(
  false_observer: list[dict[str, str]],
  truth_rows: list[dict[str, str]],
  contact_index: ContactStateIndex,
  *,
  contact_window_ms: int,
) -> list[dict[str, Any]]:
  enriched_rows: list[dict[str, Any]] = []
  for row in false_observer:
    tick_ms = base.event_tick(row)
    foot = row.get("foot", "")
    enriched: dict[str, Any] = dict(row)
    enriched.update(nearest_truth_diagnostics(truth_rows, foot, tick_ms))
    enriched.update(contact_diagnostics(contact_index, foot, tick_ms, contact_window_ms))
    enriched_rows.append(enriched)
  return enriched_rows


def match_events(
  truth_rows: list[dict[str, str]],
  observer_rows: list[dict[str, str]],
  tolerance_ms: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
  matches: list[dict[str, Any]] = []
  unmatched_observer = set(range(len(observer_rows)))

  for truth in sorted(truth_rows, key=base.event_tick):
    truth_tick = base.event_tick(truth)
    foot = truth.get("foot", "")
    best_idx: int | None = None
    best_abs_error: int | None = None

    for idx in sorted(unmatched_observer):
      obs = observer_rows[idx]
      if obs.get("foot", "") != foot:
        continue
      error = base.event_tick(obs) - truth_tick
      abs_error = abs(error)
      if abs_error > tolerance_ms:
        continue
      if best_abs_error is None or abs_error < best_abs_error:
        best_idx = idx
        best_abs_error = abs_error

    if best_idx is None:
      continue

    observer = observer_rows[best_idx]
    unmatched_observer.remove(best_idx)
    signed_error_ms = base.event_tick(observer) - truth_tick
    matches.append(
      {
        "foot": foot,
        "truth_tick_ms": truth_tick,
        "observer_tick_ms": base.event_tick(observer),
        "signed_error_ms": signed_error_ms,
        "abs_error_ms": abs(signed_error_ms),
        "truth_time_s": base.event_time(truth),
        "observer_time_s": base.event_time(observer),
        "observer_probability": observer_probability(observer),
        "observer_threshold": base.to_float(observer, "threshold"),
        "truth_normal_force": base.to_float(truth, "normal_force"),
        "truth_contact_count": base.to_int(truth, "contact_count"),
        "truth_other_geom_name": truth.get("other_geom_name", ""),
      }
    )

  matched_truth_keys = {(match["foot"], match["truth_tick_ms"]) for match in matches}
  missed_truth = [
    row
    for row in truth_rows
    if (row.get("foot", ""), base.event_tick(row)) not in matched_truth_keys
  ]
  false_observer = [observer_rows[idx] for idx in sorted(unmatched_observer)]
  return matches, missed_truth, false_observer


def missed_streaks(
  truth_rows: list[dict[str, str]],
  missed_truth: list[dict[str, str]],
  foot: str | None = None,
) -> list[dict[str, Any]]:
  missed_keys = {(row.get("foot", ""), base.event_tick(row)) for row in missed_truth}
  rows = sorted(
    [row for row in truth_rows if foot is None or row.get("foot", "") == foot],
    key=base.event_tick,
  )

  streaks: list[dict[str, Any]] = []
  current: list[dict[str, str]] = []
  for row in rows:
    key = (row.get("foot", ""), base.event_tick(row))
    if key in missed_keys:
      current.append(row)
      continue
    if current:
      streaks.append(streak_summary(current))
      current = []
  if current:
    streaks.append(streak_summary(current))
  return streaks


def streak_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
  ticks = [base.event_tick(row) for row in rows]
  feet = [row.get("foot", "") for row in rows]
  return {
    "length": len(rows),
    "start_tick_ms": min(ticks),
    "end_tick_ms": max(ticks),
    "duration_ms": max(ticks) - min(ticks),
    "feet": " ".join(feet),
    "start_time_s": min(base.event_time(row) for row in rows),
    "end_time_s": max(base.event_time(row) for row in rows),
  }


def summarize(
  truth_rows: list[dict[str, str]],
  observer_rows: list[dict[str, str]],
  matches: list[dict[str, Any]],
  missed_truth: list[dict[str, str]],
  false_observer: list[dict[str, Any]],
  tolerance_ms: int,
) -> dict[str, Any]:
  errors = [float(match["signed_error_ms"]) for match in matches]
  abs_errors = [float(match["abs_error_ms"]) for match in matches]
  probabilities = [float(match["observer_probability"]) for match in matches]
  all_streaks = missed_streaks(truth_rows, missed_truth)
  left_streaks = missed_streaks(truth_rows, missed_truth, "left")
  right_streaks = missed_streaks(truth_rows, missed_truth, "right")
  worst = max(all_streaks, key=lambda row: row["length"], default=None)
  truth_count = len(truth_rows)
  observer_count = len(observer_rows)
  matched = len(matches)
  return {
    "tolerance_ms": tolerance_ms,
    "truth_events": truth_count,
    "observer_touchdown_events": observer_count,
    "matched_events": matched,
    "missed_truth_events": len(missed_truth),
    "false_observer_events": len(false_observer),
    "precision": matched / observer_count if observer_count else None,
    "recall": matched / truth_count if truth_count else None,
    "signed_error_ms_mean": mean(errors) if errors else None,
    "signed_error_ms_median": median(errors) if errors else None,
    "abs_error_ms_mean": mean(abs_errors) if abs_errors else None,
    "abs_error_ms_median": median(abs_errors) if abs_errors else None,
    "abs_error_ms_p90": percentile(abs_errors, 0.90),
    "matched_probability_mean": mean(probabilities) if probabilities else None,
    "matched_probability_p10": percentile(probabilities, 0.10),
    "truth_by_foot": {
      foot: sum(1 for row in truth_rows if row.get("foot") == foot)
      for foot in ("left", "right")
    },
    "observer_by_foot": {
      foot: sum(1 for row in observer_rows if row.get("foot") == foot)
      for foot in ("left", "right")
    },
    "matched_by_foot": {
      foot: sum(1 for row in matches if row.get("foot") == foot)
      for foot in ("left", "right")
    },
    "missed_by_foot": {
      foot: sum(1 for row in missed_truth if row.get("foot") == foot)
      for foot in ("left", "right")
    },
    "false_positive_contact_any_within_window": sum(
      int(row.get("contact_any_within_window", 0) or 0)
      for row in false_observer
    ),
    "false_positive_no_contact_within_window": sum(
      1
      for row in false_observer
      if not int(row.get("contact_any_within_window", 0) or 0)
    ),
    "missed_streak_count": len(all_streaks),
    "max_consecutive_missed": worst["length"] if worst else 0,
    "max_consecutive_missed_left": max(
      (row["length"] for row in left_streaks),
      default=0,
    ),
    "max_consecutive_missed_right": max(
      (row["length"] for row in right_streaks),
      default=0,
    ),
    "worst_missed_streak": worst,
  }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
  fieldnames: list[str] = []
  seen: set[str] = set()
  for row in rows:
    for key in row:
      if key not in seen:
        seen.add(key)
        fieldnames.append(key)
  if not fieldnames:
    fieldnames = ["empty"]
  with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  g1_dir = Path(__file__).resolve().parents[1]
  default_log_dir = base.latest_run_dir(g1_dir / "log")

  parser = argparse.ArgumentParser()
  parser.add_argument("--log-dir", type=Path, default=default_log_dir)
  parser.add_argument("--truth", type=Path, default=None)
  parser.add_argument("--observer", type=Path, default=None)
  parser.add_argument("--event-type", default="observer_touchdown")
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--frame-log", type=Path, default=None)
  parser.add_argument("--tolerance-ms", type=int, default=80)
  parser.add_argument("--eval-start-tick-ms", type=int, default=None)
  parser.add_argument("--eval-end-tick-ms", type=int, default=None)
  parser.add_argument("--no-frame-window", action="store_true")
  parser.add_argument("--include-stationary", action="store_true")
  parser.add_argument("--min-command-norm", type=float, default=0.05)
  parser.add_argument("--moving-merge-gap-ms", type=int, default=100)
  parser.add_argument("--moving-pad-ms", type=int, default=100)
  parser.add_argument("--contact-window-ms", type=int, default=50)
  args = parser.parse_args()

  truth_path = args.truth or args.log_dir / "touchdown_truth.csv"
  observer_path = args.observer or args.log_dir / "foot_event_observer_events.csv"
  frame_log_path = args.frame_log or args.log_dir / "mjlab_g1_touchdown_log.csv"
  out_dir = args.out_dir or args.log_dir

  truth_file_rows = base.read_rows(truth_path)
  raw_truth_rows = [
    row for row in truth_file_rows if row.get("event_type") == "sim_touchdown"
  ]
  contact_rows = [
    row for row in truth_file_rows if row.get("event_type") == "contact_state"
  ]
  raw_observer_rows = [
    row for row in base.read_rows(observer_path) if row.get("event_type") == args.event_type
  ]
  contact_tick_offset_ms = infer_contact_tick_offset(contact_rows, frame_log_path)
  contact_index = ContactStateIndex(
    contact_rows,
    tick_offset_ms=contact_tick_offset_ms,
  )

  eval_start_tick_ms = args.eval_start_tick_ms
  eval_end_tick_ms = args.eval_end_tick_ms
  if not args.no_frame_window:
    frame_start, frame_end = base.frame_tick_window(frame_log_path)
    if eval_start_tick_ms is None:
      eval_start_tick_ms = frame_start
    if eval_end_tick_ms is None:
      eval_end_tick_ms = frame_end

  truth_rows = base.filter_by_tick_window(
    raw_truth_rows,
    eval_start_tick_ms,
    eval_end_tick_ms,
  )
  observer_rows = base.filter_by_tick_window(
    raw_observer_rows,
    eval_start_tick_ms,
    eval_end_tick_ms,
  )

  moving_windows: list[tuple[int, int]] = []
  if not args.include_stationary:
    moving_windows = base.moving_tick_windows(
      frame_log_path,
      args.min_command_norm,
      args.moving_merge_gap_ms,
      args.moving_pad_ms,
    )
    truth_rows = base.filter_by_tick_windows(truth_rows, moving_windows)
    observer_rows = base.filter_by_tick_windows(observer_rows, moving_windows)

  out_dir.mkdir(parents=True, exist_ok=True)
  matches, missed_truth, false_observer = match_events(
    truth_rows,
    observer_rows,
    args.tolerance_ms,
  )
  enriched_false_observer = enrich_false_observer(
    false_observer,
    truth_rows,
    contact_index,
    contact_window_ms=args.contact_window_ms,
  )
  summary = summarize(
    truth_rows,
    observer_rows,
    matches,
    missed_truth,
    enriched_false_observer,
    args.tolerance_ms,
  )
  summary.update(
    {
      "raw_truth_events": len(raw_truth_rows),
      "raw_contact_state_rows": len(contact_rows),
      "raw_observer_events": len(raw_observer_rows),
      "event_type": args.event_type,
      "contact_window_ms": args.contact_window_ms,
      "contact_tick_offset_ms": contact_tick_offset_ms,
      "eval_start_tick_ms": eval_start_tick_ms,
      "eval_end_tick_ms": eval_end_tick_ms,
      "truth_csv": str(truth_path),
      "observer_csv": str(observer_path),
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

  matches_path = out_dir / "foot_event_observer_matches.csv"
  missed_path = out_dir / "foot_event_observer_missed_truth.csv"
  false_path = out_dir / "foot_event_observer_false_positive.csv"
  streaks_path = out_dir / "foot_event_observer_missed_streaks.csv"
  summary_json_path = out_dir / "foot_event_observer_summary.json"
  summary_txt_path = out_dir / "foot_event_observer_summary.txt"

  write_rows(matches_path, matches)
  write_rows(missed_path, missed_truth)
  write_rows(false_path, enriched_false_observer)
  write_rows(streaks_path, missed_streaks(truth_rows, missed_truth))
  with summary_json_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
    f.write("\n")

  lines = [
    "Foot event observer passive validation",
    f"log_dir: {args.log_dir}",
    f"truth_csv: {truth_path}",
    f"observer_csv: {observer_path}",
    f"event_type: {args.event_type}",
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
      f"contact_state={summary['raw_contact_state_rows']} "
      f"observer={summary['raw_observer_events']}"
    ),
    (
      "contact_diagnostics: "
      f"window_ms={summary['contact_window_ms']} "
      f"tick_offset_ms={summary['contact_tick_offset_ms']} "
      f"fp_contact_any={summary['false_positive_contact_any_within_window']} "
      f"fp_no_contact={summary['false_positive_no_contact_within_window']}"
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
      f"observer={summary['observer_touchdown_events']} "
      f"matched={summary['matched_events']} "
      f"missed={summary['missed_truth_events']} "
      f"false_positive={summary['false_observer_events']}"
    ),
    (f"metrics: precision={summary['precision']} recall={summary['recall']}"),
    (
      "timing_ms: "
      f"signed_median={summary['signed_error_ms_median']} "
      f"abs_median={summary['abs_error_ms_median']} "
      f"abs_p90={summary['abs_error_ms_p90']}"
    ),
    (
      "miss_streaks: "
      f"max={summary['max_consecutive_missed']} "
      f"left={summary['max_consecutive_missed_left']} "
      f"right={summary['max_consecutive_missed_right']} "
      f"count={summary['missed_streak_count']}"
    ),
    f"truth_by_foot: {summary['truth_by_foot']}",
    f"observer_by_foot: {summary['observer_by_foot']}",
    f"matched_by_foot: {summary['matched_by_foot']}",
    f"missed_by_foot: {summary['missed_by_foot']}",
    f"matches_csv: {matches_path}",
    f"missed_csv: {missed_path}",
    f"false_positive_csv: {false_path}",
    f"streaks_csv: {streaks_path}",
    f"summary_json: {summary_json_path}",
  ]
  with summary_txt_path.open("w", encoding="utf-8") as f:
    f.write("\n".join(lines))
    f.write("\n")
  print("\n".join(lines))


if __name__ == "__main__":
  main()
