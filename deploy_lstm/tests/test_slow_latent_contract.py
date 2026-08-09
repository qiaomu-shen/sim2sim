"""Static deployment-contract checks for the G1 slow-latent policy."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "robots/g1/config/policy/velocity/slow_latent"
MODEL_PATH = POLICY_DIR / "exported/policy.onnx"
DEPLOY_PATH = POLICY_DIR / "params/deploy.yaml"
MJLAB_ROOT = Path(os.environ.get("MJLAB_ROOT", "/home/ubt2204/work/mjlab"))


def resolve_repo_path(path: str) -> Path:
  candidate = Path(path)
  if candidate.is_absolute():
    return candidate
  for base in (ROOT, ROOT.parent):
    resolved = base / candidate
    if resolved.exists():
      return resolved
  return ROOT / candidate


def foot_event_fixture() -> list[dict[str, object]]:
  probe_steps = [
    ("footprint", 0, 0.00, 0.00, None),
    ("release", -1, 0.00, 0.00, None),
    ("release", -1, 0.00, 0.00, None),
    ("toe", 0, 0.20, 0.00, "probe_first_hit"),
    ("footprint", 1, 0.25, 0.15, None),
    ("release", -1, 0.00, 0.00, None),
    ("release", -1, 0.00, 0.00, None),
    ("footprint", 0, 0.50, 0.30, "probe_bootstrap"),
    ("release", -1, 0.00, 0.00, None),
    ("release", -1, 0.00, 0.00, None),
    ("footprint", 1, 0.76, 0.45, "probe_small_1"),
    ("release", -1, 0.00, 0.00, None),
    ("release", -1, 0.00, 0.00, None),
    ("footprint", 0, 1.02, 0.60, "probe_small_2"),
    ("release", -1, 0.00, 0.00, None),
    ("release", -1, 0.00, 0.00, None),
    ("footprint", 1, 1.35, 0.75, None),
    ("release", -1, 0.00, 0.00, None),
    ("release", -1, 0.00, 0.00, None),
    ("footprint", 0, 1.65, 0.90, None),
    ("release", -1, 0.00, 0.00, None),
    ("release", -1, 0.00, 0.00, None),
    ("footprint", 1, 2.03, 1.05, None),
    ("release", -1, 0.00, 0.00, None),
    ("release", -1, 0.00, 0.00, None),
    ("footprint", 0, 2.38, 1.20, "probe_cap_not_reached"),
  ]
  lock_steps = [
    ("footprint", 0, 0.00, 0.00, None),
    ("release", -1, 0.00, 0.00, None),
    ("release", -1, 0.00, 0.00, None),
    ("toe", 0, 0.20, 0.00, "lock_first_hit"),
    ("footprint", 1, 0.25, 0.15, None),
    ("release", -1, 0.00, 0.00, None),
    ("release", -1, 0.00, 0.00, None),
    ("footprint", 0, 0.50, 0.30, "lock_bootstrap"),
    ("release", -1, 0.00, 0.00, None),
    ("release", -1, 0.00, 0.00, None),
    ("toe", 0, 0.70, 0.30, "lock_second_hit"),
    ("footprint", 1, 0.73, 0.45, "lock_recovery"),
    ("release", -1, 0.00, 0.00, None),
    ("release", -1, 0.00, 0.00, None),
    ("toe", 1, 1.22, 0.45, "lock_reopen"),
  ]
  events: list[dict[str, object]] = [
    {"kind": "reset"},
    {"kind": "noop", "label": "zero", "dt": 0.0},
  ]
  for kind, foot, x, z, label in probe_steps:
    event = {"kind": kind, "foot": foot, "x": x, "z": z}
    if label is not None:
      event["label"] = label
    events.append(event)
  events.append({"kind": "reset"})
  for kind, foot, x, z, label in lock_steps:
    event = {"kind": kind, "foot": foot, "x": x, "z": z}
    if label is not None:
      event["label"] = label
    events.append(event)
  events.append({"kind": "reset"})
  events.append({"kind": "noop", "label": "after_reset", "dt": 0.0})
  events.append({"kind": "reset"})
  events.append(
    {"kind": "predicted_footprint", "foot": 0, "x": 0.00, "z": 0.00}
  )
  events.append(
    {"kind": "footprint", "foot": 1, "x": 0.25, "z": 0.15, "label": "predicted_pair"}
  )
  return events


def run_cpp_foot_event_memory_fixture() -> dict[str, np.ndarray]:
  event_lines: list[str] = []
  for event in foot_event_fixture():
    kind = event["kind"]
    if kind == "reset":
      event_lines.append("      memory.configure(cfg);")
      continue
    label = event.get("label")
    label_arg = f"\"{label}\"" if label is not None else "nullptr"
    if kind == "noop":
      dt = float(event.get("dt", 0.02))
      event_lines.append(f"      noop(memory, {dt:.8f}f, {label_arg});")
    elif kind == "release":
      event_lines.append(f"      release(memory, {label_arg});")
    elif kind in {"footprint", "predicted_footprint", "toe"}:
      foot = int(event["foot"])
      x = float(event["x"])
      z = float(event["z"])
      event_lines.append(
        f"      {kind}(memory, {foot}, {x:.8f}f, {z:.8f}f, {label_arg});"
      )
    else:
      raise AssertionError(f"unknown fixture event kind: {kind}")

  source = r"""
    #include "foot_event_memory.h"

    #include <algorithm>
    #include <array>
    #include <cmath>
    #include <iostream>
    #include <string>
    #include <vector>

    using mjlab::diagnostics::FootEventMemory;
    using mjlab::diagnostics::FootEventMemoryConfig;
    using mjlab::diagnostics::FootEventMemoryFrame;

    static FootEventMemoryFrame make_frame(float dt = 0.02f) {
      FootEventMemoryFrame frame;
      frame.dt = dt;
      frame.root_quat_w = Eigen::Quaternionf::Identity();
      frame.command = {0.5f, 0.0f, 0.0f};
      frame.touchdown_prob = {1.0f, 1.0f};
      frame.toe_riser_prob = {1.0f, 1.0f};
      return frame;
    }

    static void set_foot(FootEventMemoryFrame& frame, int foot, float x, float z) {
      const float y = foot == 0 ? 0.10f : -0.10f;
      if (foot == 0) {
        frame.left_toe_pos_b = Eigen::Vector3f(x, y, z);
        frame.left_heel_pos_b = Eigen::Vector3f(x, y, z);
      } else {
        frame.right_toe_pos_b = Eigen::Vector3f(x, y, z);
        frame.right_heel_pos_b = Eigen::Vector3f(x, y, z);
      }
    }

    static void print_values(const char* name, const std::vector<float>& values) {
      if (name == nullptr) {
        return;
      }
      std::cout << name;
      for (const float value : values) {
        std::cout << ' ' << value;
      }
      std::cout << '\n';
    }

    static void noop(FootEventMemory& memory, float dt, const char* label) {
      FootEventMemoryFrame frame = make_frame(dt);
      frame.contact_prob = {0.0f, 0.0f};
      frame.touchdown_event = {false, false};
      frame.toe_riser_event = {false, false};
      print_values(label, memory.update(frame));
    }

    static void release(FootEventMemory& memory, const char* label) {
      noop(memory, 0.02f, label);
    }

    static void footprint(
        FootEventMemory& memory,
        int foot,
        float x,
        float z,
        const char* label) {
      FootEventMemoryFrame frame = make_frame();
      set_foot(frame, foot, x, z);
      frame.contact_prob = foot == 0
          ? std::array<float, 2>{1.0f, 0.0f}
          : std::array<float, 2>{0.0f, 1.0f};
      frame.touchdown_event = foot == 0
          ? std::array<bool, 2>{true, false}
          : std::array<bool, 2>{false, true};
      print_values(label, memory.update(frame));
    }

    static void predicted_footprint(
        FootEventMemory& memory,
        int foot,
        float x,
        float z,
        const char* label) {
      std::vector<float> values;
      for (int i = 0; i < 3; ++i) {
        FootEventMemoryFrame frame = make_frame();
        frame.phase = foot == 0
            ? 0.03f * static_cast<float>(i)
            : std::fmod(0.5f + 0.03f * static_cast<float>(i), 1.0f);
        set_foot(frame, foot, x, z);
        frame.contact_prob = foot == 0
            ? std::array<float, 2>{1.0f, 0.0f}
            : std::array<float, 2>{0.0f, 1.0f};
        values = memory.update(frame);
      }
      print_values(label, values);
    }

    static void toe(
        FootEventMemory& memory,
        int foot,
        float x,
        float z,
        const char* label) {
      FootEventMemoryFrame frame = make_frame();
      set_foot(frame, foot, x, z);
      frame.contact_prob = {0.0f, 0.0f};
      frame.toe_riser_event = foot == 0
          ? std::array<bool, 2>{true, false}
          : std::array<bool, 2>{false, true};
      print_values(label, memory.update(frame));
    }

    int main() {
      FootEventMemoryConfig cfg;
      cfg.memory_len = 6;
      cfg.age_norm_s = 10.0f;
      cfg.stance_age_norm_s = 10.0f;
      cfg.predicted_fill_enabled = true;
      cfg.predicted_fill_grace_frames = 3;
      cfg.predicted_fill_contact_threshold = 0.35f;
      cfg.predicted_fill_confidence = 0.35f;
      cfg.predicted_fill_touchdown_prob = 0.35f;
      cfg.touchdown_gate_enabled = false;
      FootEventMemory memory;
__EVENT_LINES__
      return 0;
    }
  """.replace("__EVENT_LINES__", "\n".join(event_lines))
  with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    source_path = tmp_path / "foot_event_memory_fixture.cpp"
    binary_path = tmp_path / "foot_event_memory_fixture"
    source_path.write_text(textwrap.dedent(source))
    subprocess.run(
      [
        "g++",
        "-std=c++17",
        "-O2",
        "-I",
        str(ROOT / "include"),
        "-I",
        "/usr/include/eigen3",
        str(source_path),
        "-o",
        str(binary_path),
      ],
      check=True,
    )
    output = subprocess.run(
      [str(binary_path)],
      check=True,
      text=True,
      capture_output=True,
    ).stdout

  parsed: dict[str, np.ndarray] = {}
  for line in output.strip().splitlines():
    name, _, values = line.partition(" ")
    parsed[name] = np.fromstring(values, sep=" ", dtype=np.float32)
  return parsed


def run_python_foot_event_memory_fixture(
  memory_cfg: dict[str, object],
) -> dict[str, np.ndarray]:
  script = r"""
import json
import sys
from types import SimpleNamespace

import torch

from mjlab.tasks.velocity.mdp.observations import FootEventMemoryObs


events = json.loads(sys.argv[1])
params = json.loads(sys.argv[2])
params.update(
  {
    "memory_len": 6,
    "age_norm_s": 10.0,
    "stance_age_norm_s": 10.0,
    "noise_enabled": False,
    "include_summary": True,
    "include_raw_memory": False,
    "log_summary_metrics": False,
  }
)


class FakeEnv:
  num_envs = 1
  device = "cpu"

  def __init__(self):
    self.extras = {}


env = FakeEnv()
memory = FootEventMemoryObs(SimpleNamespace(params=params), env)
root_pos = torch.zeros(1, 3, dtype=torch.float32)
root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
phase = torch.zeros(1, 2, dtype=torch.float32)
mask = torch.ones(1, dtype=torch.bool)
zero = torch.zeros(1, dtype=torch.float32)
one = torch.ones(1, dtype=torch.float32)
outputs = {}
predicted_confidence = torch.full(
  (1,),
  float(params.get("predicted_fill_confidence", 0.35)),
  dtype=torch.float32,
)
predicted_touchdown_prob = torch.full(
  (1,),
  float(params.get("predicted_fill_touchdown_prob", 0.35)),
  dtype=torch.float32,
)


def point(foot, x, z):
  y = 0.10 if foot == 0 else -0.10
  return torch.tensor([[float(x), y, float(z)]], dtype=torch.float32)


def summarize(label, new_footprint=False, new_toe=False, dt=0.02):
  summary = memory._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([new_footprint], dtype=torch.bool),
    new_toe_mark_any=torch.tensor([new_toe], dtype=torch.bool),
    step_dt=float(dt),
  )[0].detach().cpu().numpy()
  if label is not None:
    outputs[label] = summary.tolist()


for event in events:
  kind = event["kind"]
  if kind == "reset":
    memory.reset()
    continue
  dt = float(event.get("dt", 0.02))
  memory._age_and_refresh(dt, root_pos, root_quat)
  label = event.get("label")
  if kind in ("noop", "release"):
    summarize(label, False, False, dt)
  elif kind == "footprint":
    foot = int(event["foot"])
    memory._push_footprint(
      mask=mask,
      foot_id=foot,
      point_body=point(foot, event["x"], event["z"]),
      root_pos_w=root_pos,
      root_quat_w=root_quat,
      phase_now=phase,
      contact_prob=one,
      touchdown_prob=one,
      confidence=one,
      source_predicted_fill=False,
    )
    summarize(label, True, False, dt)
  elif kind == "predicted_footprint":
    foot = int(event["foot"])
    memory._push_footprint(
      mask=mask,
      foot_id=foot,
      point_body=point(foot, event["x"], event["z"]),
      root_pos_w=root_pos,
      root_quat_w=root_quat,
      phase_now=phase,
      contact_prob=one,
      touchdown_prob=predicted_touchdown_prob,
      confidence=predicted_confidence,
      source_predicted_fill=True,
    )
    summarize(label, True, False, dt)
  elif kind == "toe":
    foot = int(event["foot"])
    memory._push_toe_mark(
      mask=mask,
      foot_id=foot,
      point_body=point(foot, event["x"], event["z"]),
      root_pos_w=root_pos,
      root_quat_w=root_quat,
      phase_now=phase,
      contact_prob=zero,
      toe_prob=one,
      confidence=one,
      swing_or_early_contact=mask,
      false_positive=False,
    )
    summarize(label, False, True, dt)
  else:
    raise RuntimeError(f"unknown event kind {kind}")

print(json.dumps(outputs))
  """
  result = subprocess.run(
    [
      "uv",
      "run",
      "python",
      "-c",
      script,
      json.dumps(foot_event_fixture()),
      json.dumps(memory_cfg),
    ],
    cwd=MJLAB_ROOT,
    check=True,
    text=True,
    capture_output=True,
  )
  parsed = json.loads(result.stdout)
  return {
    name: np.asarray(values, dtype=np.float32)
    for name, values in parsed.items()
  }


def run_cpp_foot_odometry_fixture() -> np.ndarray:
  source = r"""
    #define MJLAB_FOOT_EVENT_MEMORY_TESTING
    #include "foot_event_memory.h"

    #include <array>
    #include <cmath>
    #include <iostream>
    #include <vector>

    using mjlab::diagnostics::FootEventMemory;
    using mjlab::diagnostics::FootEventMemoryConfig;
    using mjlab::diagnostics::FootEventMemoryFrame;

    static Eigen::Quaternionf yaw_quat(float yaw) {
      return Eigen::Quaternionf(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
    }

    static Eigen::Vector3f body_from_world(
        const Eigen::Vector3f& base,
        const Eigen::Quaternionf& quat,
        const Eigen::Vector3f& point_w) {
      return quat.conjugate() * (point_w - base);
    }

    static FootEventMemoryFrame make_frame(
        const Eigen::Vector3f& base,
        const Eigen::Quaternionf& quat,
        const Eigen::Vector3f& left_w,
        const Eigen::Vector3f& right_w) {
      FootEventMemoryFrame frame;
      frame.dt = 0.02f;
      frame.root_quat_w = quat;
      frame.command = {0.5f, 0.0f, 0.0f};
      frame.touchdown_prob = {1.0f, 1.0f};
      frame.toe_riser_prob = {1.0f, 1.0f};
      frame.left_toe_pos_b = body_from_world(base, quat, left_w);
      frame.left_heel_pos_b = frame.left_toe_pos_b;
      frame.right_toe_pos_b = body_from_world(base, quat, right_w);
      frame.right_heel_pos_b = frame.right_toe_pos_b;
      return frame;
    }

    int main() {
      FootEventMemoryConfig cfg;
      cfg.memory_len = 6;
      cfg.age_norm_s = 10.0f;
      cfg.stance_age_norm_s = 10.0f;
      cfg.odom_max_double_support_residual_m = 0.06f;
      cfg.touchdown_gate_enabled = false;
      FootEventMemory memory;
      memory.configure(cfg);

      const Eigen::Quaternionf q0 = yaw_quat(0.0f);
      const Eigen::Quaternionf q90 = yaw_quat(0.5f * static_cast<float>(M_PI));
      const Eigen::Vector3f base0(0.0f, 0.0f, 0.0f);
      const Eigen::Vector3f base_mid(0.15f, 0.0f, 0.0f);
      const Eigen::Vector3f base1(0.30f, 0.0f, 0.0f);
      const Eigen::Vector3f left0(0.0f, 0.10f, 0.0f);
      const Eigen::Vector3f right0(0.50f, -0.10f, 0.0f);

      auto frame = make_frame(base0, q0, left0, right0);
      frame.contact_prob = {1.0f, 0.0f};
      frame.touchdown_event = {true, false};
      memory.update(frame);

      frame = make_frame(base_mid, q0, left0, right0);
      frame.contact_prob = {1.0f, 0.0f};
      memory.update(frame);

      frame = make_frame(base1, q0, left0, right0);
      frame.contact_prob = {1.0f, 0.0f};
      memory.update(frame);
      const float base_after_move = memory.odom_base_pos_for_testing().x();

      frame = make_frame(base1, q0, left0, right0);
      frame.contact_prob = {1.0f, 1.0f};
      frame.touchdown_event = {false, true};
      const auto& translated_summary = memory.update(frame);
      const float translated_stride = translated_summary[5];

      frame = make_frame(base1, q0, left0, right0);
      frame.contact_prob = {1.0f, 1.0f};
      frame.left_toe_pos_b.x() = -0.40f;  // left estimate would jump to x=0.40
      frame.left_heel_pos_b = frame.left_toe_pos_b;
      memory.update(frame);
      const float base_after_single_slip = memory.odom_base_pos_for_testing().x();

      frame = make_frame(base1, q0, left0, right0);
      frame.contact_prob = {1.0f, 1.0f};
      frame.left_toe_pos_b.x() = -0.35f;   // left estimate x=0.35
      frame.left_heel_pos_b = frame.left_toe_pos_b;
      frame.right_toe_pos_b.x() = 0.25f;   // right estimate x=0.25
      frame.right_heel_pos_b = frame.right_toe_pos_b;
      memory.update(frame);
      const float base_after_ambiguous_slip = memory.odom_base_pos_for_testing().x();

      frame = make_frame(base1, q0, left0, right0);
      frame.contact_prob = {0.0f, 1.0f};
      memory.update(frame);
      memory.update(frame);

      frame = make_frame(base1, q90, left0, right0);
      frame.contact_prob = {0.0f, 1.0f};
      memory.update(frame);
      const float base_after_yaw = memory.odom_base_pos_for_testing().x();

      const Eigen::Vector3f left1(0.30f, 0.40f, 0.15f);
      frame = make_frame(base1, q90, left1, right0);
      frame.contact_prob = {1.0f, 1.0f};
      frame.touchdown_event = {true, false};
      const auto& yaw_summary = memory.update(frame);
      const float yaw_stride = yaw_summary[5];
      const float yaw_height = yaw_summary[6];

      std::cout
          << base_after_move << ' '
          << translated_stride << ' '
          << base_after_single_slip << ' '
          << base_after_ambiguous_slip << ' '
          << base_after_yaw << ' '
          << yaw_stride << ' '
          << yaw_height << '\n';
      return 0;
    }
  """
  with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    source_path = tmp_path / "foot_odometry_fixture.cpp"
    binary_path = tmp_path / "foot_odometry_fixture"
    source_path.write_text(textwrap.dedent(source))
    subprocess.run(
      [
        "g++",
        "-std=c++17",
        "-O2",
        "-I",
        str(ROOT / "include"),
        "-I",
        "/usr/include/eigen3",
        str(source_path),
        "-o",
        str(binary_path),
      ],
      check=True,
    )
    output = subprocess.run(
      [str(binary_path)],
      check=True,
      text=True,
      capture_output=True,
    ).stdout
  return np.fromstring(output, sep=" ", dtype=np.float32)


def run_cpp_predicted_fill_fixture() -> np.ndarray:
  source = r"""
    #define MJLAB_FOOT_EVENT_MEMORY_TESTING
    #include "foot_event_memory.h"

    #include <array>
    #include <cmath>
    #include <iostream>

    using mjlab::diagnostics::FootEventMemory;
    using mjlab::diagnostics::FootEventMemoryConfig;
    using mjlab::diagnostics::FootEventMemoryFrame;

    static FootEventMemoryFrame frame(float phase, float contact, bool touchdown) {
      FootEventMemoryFrame frame;
      frame.dt = 0.02f;
      frame.phase = phase;
      frame.command = {0.5f, 0.0f, 0.0f};
      frame.root_quat_w = Eigen::Quaternionf::Identity();
      frame.left_toe_pos_b = Eigen::Vector3f(0.0f, 0.10f, 0.0f);
      frame.left_heel_pos_b = frame.left_toe_pos_b;
      frame.right_toe_pos_b = Eigen::Vector3f(0.25f, -0.10f, 0.0f);
      frame.right_heel_pos_b = frame.right_toe_pos_b;
      frame.contact_prob = {contact, 0.0f};
      frame.touchdown_prob = {touchdown ? 1.0f : 0.0f, 0.0f};
      frame.touchdown_event = {touchdown, false};
      return frame;
    }

    static FootEventMemoryFrame world_frame(
        float phase,
        float base_x,
        float contact,
        bool touchdown) {
      FootEventMemoryFrame frame;
      frame.dt = 0.02f;
      frame.phase = phase;
      frame.command = {0.5f, 0.0f, 0.0f};
      frame.root_quat_w = Eigen::Quaternionf::Identity();
      const Eigen::Vector3f base(base_x, 0.0f, 0.0f);
      const Eigen::Vector3f left_w(0.0f, 0.10f, 0.0f);
      const Eigen::Vector3f right_w(0.25f, -0.10f, 0.0f);
      frame.left_toe_pos_b = left_w - base;
      frame.left_heel_pos_b = frame.left_toe_pos_b;
      frame.right_toe_pos_b = right_w - base;
      frame.right_heel_pos_b = frame.right_toe_pos_b;
      frame.contact_prob = {contact, 0.0f};
      frame.touchdown_prob = {touchdown ? 1.0f : 0.0f, 0.0f};
      frame.touchdown_event = {touchdown, false};
      return frame;
    }

    int main() {
      FootEventMemoryConfig cfg;
      cfg.memory_len = 6;
      cfg.age_norm_s = 10.0f;
      cfg.stance_age_norm_s = 10.0f;
      cfg.predicted_fill_enabled = true;
      cfg.predicted_fill_grace_frames = 3;
      cfg.predicted_fill_phase_window = 0.18f;
      cfg.predicted_fill_phase_lead_window = 0.08f;
      cfg.predicted_fill_contact_threshold = 0.35f;
      cfg.predicted_fill_confidence = 0.35f;
      cfg.predicted_fill_touchdown_prob = 0.35f;
      cfg.touchdown_gate_enabled = true;
      cfg.touchdown_gate_contact_threshold = 0.20f;
      FootEventMemory memory;
      memory.configure(cfg);

      memory.update(frame(0.00f, 1.0f, false));
      memory.update(frame(0.03f, 1.0f, false));
      const float count_before_fill =
          static_cast<float>(memory.valid_footprint_count_for_testing());
      memory.update(frame(0.06f, 1.0f, false));
      const auto predicted = memory.footprint_for_testing(0);
      const float count_after_fill =
          static_cast<float>(memory.valid_footprint_count_for_testing());
      const float pending_after_fill =
          memory.predicted_fill_pending_for_testing(0) ? 1.0f : 0.0f;
      memory.update(frame(0.09f, 1.0f, true));
      const float count_after_late_touchdown =
          static_cast<float>(memory.valid_footprint_count_for_testing());

      memory.configure(cfg);
      memory.update(frame(0.00f, 1.0f, false));
      memory.update(frame(0.03f, 1.0f, true));
      const auto confirmed = memory.footprint_for_testing(0);
      const float timely_count =
          static_cast<float>(memory.valid_footprint_count_for_testing());

      memory.configure(cfg);
      memory.update(frame(0.00f, 0.0f, false));
      memory.update(frame(0.03f, 0.0f, false));
      memory.update(frame(0.06f, 0.0f, false));
      const float unsupported_count =
          static_cast<float>(memory.valid_footprint_count_for_testing());

      memory.configure(cfg);
      memory.update(frame(0.00f, 1.0f, false));
      memory.configure(cfg);
      memory.update(frame(0.03f, 1.0f, false));
      const float reset_first_frame_count =
          static_cast<float>(memory.valid_footprint_count_for_testing());
      const float reset_pending =
          memory.predicted_fill_pending_for_testing(0) ? 1.0f : 0.0f;

      memory.configure(cfg);
      memory.update(frame(0.98f, 1.0f, false));
      const float early_count =
          static_cast<float>(memory.valid_footprint_count_for_testing());
      const float early_pending =
          memory.predicted_fill_pending_for_testing(0) ? 1.0f : 0.0f;
      memory.update(frame(0.00f, 1.0f, true));
      const auto early_confirmed = memory.footprint_for_testing(0);
      const float early_accept_count =
          static_cast<float>(memory.valid_footprint_count_for_testing());

      memory.configure(cfg);
      memory.update(world_frame(0.00f, 0.00f, 1.0f, false));
      memory.update(world_frame(0.03f, 0.10f, 1.0f, false));
      memory.update(world_frame(0.06f, 0.20f, 1.0f, false));
      const float late_odom_before = memory.odom_base_pos_for_testing().x();
      memory.update(world_frame(0.09f, 0.25f, 1.0f, true));
      const float late_odom_after = memory.odom_base_pos_for_testing().x();
      const float late_count =
          static_cast<float>(memory.valid_footprint_count_for_testing());

      memory.configure(cfg);
      memory.update(frame(0.25f, 0.0f, true));
      const float false_positive_count =
          static_cast<float>(memory.valid_footprint_count_for_testing());

      std::cout
          << count_before_fill << ' '
          << count_after_fill << ' '
          << predicted[3] << ' '
          << predicted[4] << ' '
          << predicted[5] << ' '
          << predicted[15] << ' '
          << pending_after_fill << ' '
          << count_after_late_touchdown << ' '
          << timely_count << ' '
          << confirmed[3] << ' '
          << confirmed[4] << ' '
          << unsupported_count << ' '
          << reset_first_frame_count << ' '
          << reset_pending << ' '
          << early_count << ' '
          << early_pending << ' '
          << early_accept_count << ' '
          << early_confirmed[3] << ' '
          << early_confirmed[4] << ' '
          << late_odom_before << ' '
          << late_odom_after << ' '
          << late_count << ' '
          << false_positive_count << '\n';
      return 0;
    }
  """
  with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    source_path = tmp_path / "predicted_fill_fixture.cpp"
    binary_path = tmp_path / "predicted_fill_fixture"
    source_path.write_text(textwrap.dedent(source))
    subprocess.run(
      [
        "g++",
        "-std=c++17",
        "-O2",
        "-I",
        str(ROOT / "include"),
        "-I",
        "/usr/include/eigen3",
        str(source_path),
        "-o",
        str(binary_path),
      ],
      check=True,
    )
    output = subprocess.run(
      [str(binary_path)],
      check=True,
      text=True,
      capture_output=True,
    ).stdout
  return np.fromstring(output, sep=" ", dtype=np.float32)


def run_cpp_ort_fail_safe_fixture() -> tuple[int, bool, int, int]:
  source = r"""
    #define MJLAB_ORT_RUNNER_TESTING
    #include "isaaclab/algorithms/algorithms.h"

    #include <cmath>
    #include <iostream>
    #include <limits>
    #include <string>
    #include <unordered_map>
    #include <vector>

    int main(int argc, char** argv) {
      if (argc != 2) {
        return 2;
      }
      isaaclab::OrtRunner runner(argv[1]);
      std::unordered_map<std::string, std::vector<float>> obs;
      obs["actor_obs"] = std::vector<float>(490, 0.0f);
      obs["latent_obs"] = std::vector<float>(173, 0.0f);
      runner.act(obs);
      const size_t nonzero_before = runner.recurrent_nonzero_count_for_testing();
      obs["actor_obs"][17] = std::numeric_limits<float>::quiet_NaN();
      const auto action = runner.act(obs);
      const size_t nonzero_after = runner.recurrent_nonzero_count_for_testing();
      bool all_zero = action.size() == 29;
      for (const float value : action) {
        all_zero = all_zero && std::isfinite(value) && std::abs(value) <= 1.0e-8f;
      }
      std::cout
          << action.size() << ' '
          << (all_zero ? 1 : 0) << ' '
          << nonzero_before << ' '
          << nonzero_after << '\n';
      return all_zero && nonzero_before > 0 && nonzero_after == 0 ? 0 : 1;
    }
  """
  with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    source_path = tmp_path / "ort_fail_safe_fixture.cpp"
    binary_path = tmp_path / "ort_fail_safe_fixture"
    source_path.write_text(textwrap.dedent(source))
    ort_dir = ROOT / "thirdparty/onnxruntime-linux-x64-1.22.0"
    ort_lib = ort_dir / "lib/libonnxruntime.so.1.22.0"
    subprocess.run(
      [
        "g++",
        "-std=c++17",
        "-O2",
        "-I",
        str(ROOT / "include"),
        "-I",
        str(ort_dir / "include"),
        str(source_path),
        str(ort_lib),
        "-Wl,-rpath," + str(ort_dir / "lib"),
        "-o",
        str(binary_path),
      ],
      check=True,
    )
    output = subprocess.run(
      [str(binary_path), str(MODEL_PATH)],
      check=True,
      text=True,
      capture_output=True,
    ).stdout.strip()
  size_s, all_zero_s, before_s, after_s = output.split()
  return int(size_s), all_zero_s == "1", int(before_s), int(after_s)


def run_cpp_observer_reset_fixture(
  footprint_path: Path,
  toe_path: Path,
) -> np.ndarray:
  source = r"""
    #define MJLAB_FOOT_EVENT_OBSERVER_TESTING
    #include "foot_event_observer.h"

    #include <cmath>
    #include <filesystem>
    #include <iostream>
    #include <vector>

    using mjlab::diagnostics::FootEventObserver;
    using mjlab::diagnostics::FootEventObserverConfig;
    using mjlab::diagnostics::FootEventObserverFrame;

    static FootEventObserverFrame make_frame(int step, float offset) {
      FootEventObserverFrame frame;
      frame.state = "test";
      frame.step = step;
      frame.time_s = 1.0 + 0.02 * step;
      frame.phase = 0.25f;
      frame.command = {0.5f, 0.0f, 0.0f};
      frame.root_quat_w = Eigen::Quaternionf::Identity();
      frame.projected_gravity_b = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
      frame.root_ang_vel_b = Eigen::Vector3f(0.1f + offset, 0.2f, 0.3f);
      frame.left_toe_pos_b = Eigen::Vector3f(0.2f + offset, 0.1f, -0.7f);
      frame.right_toe_pos_b = Eigen::Vector3f(-0.2f, -0.1f, -0.7f);
      frame.left_heel_pos_b = Eigen::Vector3f(0.1f + offset, 0.1f, -0.72f);
      frame.right_heel_pos_b = Eigen::Vector3f(-0.3f, -0.1f, -0.72f);
      frame.prev_action.assign(29, 0.1f);
      frame.action_scale.assign(29, 0.2f);
      frame.joint_pos = Eigen::VectorXf::Constant(29, 0.3f + offset);
      frame.default_joint_pos = Eigen::VectorXf::Constant(29, 0.3f);
      frame.joint_vel = Eigen::VectorXf::Constant(29, 0.4f + offset);
      return frame;
    }

    static float delta_norm(const std::vector<float>& features) {
      float total = 0.0f;
      const int ranges[][2] = {{6, 9}, {30, 42}, {78, 90}};
      for (const auto& range : ranges) {
        for (int i = range[0]; i < range[1]; ++i) {
          total += std::abs(features.at(static_cast<size_t>(i)));
        }
      }
      return total;
    }

    int main(int argc, char** argv) {
      if (argc != 4) {
        return 2;
      }
      FootEventObserverConfig cfg;
      cfg.enabled = true;
      cfg.feature_dim = 93;
      cfg.history_frames = 24;
      cfg.event_path = argv[3];
      cfg.footprint_touchdown_detector.name = "footprint_touchdown_detector";
      cfg.footprint_touchdown_detector.model_path = argv[1];
      cfg.toe_riser_detector.name = "toe_riser_detector";
      cfg.toe_riser_detector.model_path = argv[2];

      FootEventObserver observer;
      observer.configure(cfg);
      if (!observer.open()) {
        return 3;
      }
      observer.update(make_frame(1, 0.0f));
      observer.update(make_frame(2, 0.05f));
      const float history_before_reset =
          static_cast<float>(observer.history_size_for_testing());
      const float delta_before_reset =
          delta_norm(observer.last_features_for_testing());
      observer.reset();
      observer.update(make_frame(3, 0.10f));
      const float history_after_reset =
          static_cast<float>(observer.history_size_for_testing());
      const float delta_after_reset =
          delta_norm(observer.last_features_for_testing());
      std::cout
          << history_before_reset << ' '
          << delta_before_reset << ' '
          << history_after_reset << ' '
          << delta_after_reset << '\n';
      return 0;
    }
  """
  with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    source_path = tmp_path / "observer_reset_fixture.cpp"
    binary_path = tmp_path / "observer_reset_fixture"
    log_path = tmp_path / "observer_events.csv"
    source_path.write_text(textwrap.dedent(source))
    ort_dir = ROOT / "thirdparty/onnxruntime-linux-x64-1.22.0"
    ort_lib = ort_dir / "lib/libonnxruntime.so.1.22.0"
    subprocess.run(
      [
        "g++",
        "-std=c++17",
        "-O2",
        "-I",
        str(ROOT / "include"),
        "-I",
        str(ort_dir / "include"),
        "-I",
        "/usr/include/eigen3",
        str(source_path),
        str(ort_lib),
        "-Wl,-rpath," + str(ort_dir / "lib"),
        "-o",
        str(binary_path),
      ],
      check=True,
    )
    output = subprocess.run(
      [str(binary_path), str(footprint_path), str(toe_path), str(log_path)],
      check=True,
      text=True,
      capture_output=True,
    ).stdout
  return np.fromstring(output, sep=" ", dtype=np.float32)


def run_cpp_manager_reset_fixture() -> np.ndarray:
  source = r"""
    #include "isaaclab/envs/manager_based_rl_env.h"
    #include "isaaclab/manager/action_manager.h"
    #include "isaaclab/manager/observation_manager.h"

    #include <algorithm>
    #include <iostream>
    #include <memory>
    #include <numeric>
    #include <vector>

    namespace isaaclab {

    class ResetTestAction : public ActionTerm {
    public:
      ResetTestAction(YAML::Node cfg, ManagerBasedRLEnv* env)
      : ActionTerm(cfg, env) {}
      int action_dim() override { return 1; }
      std::vector<float> raw_actions() override { return {raw_}; }
      std::vector<float> processed_actions() override { return {raw_}; }
      void process_actions(std::vector<float> actions) override { raw_ = actions.at(0); }
      void reset() override { raw_ = 0.0f; }
    private:
      float raw_ = 0.0f;
    };
    REGISTER_ACTION(ResetTestAction)

    REGISTER_OBSERVATION(reset_test_obs)
    {
      return env->robot->data.foot_event_summary;
    }

    class ResetTestAlg : public Algorithms {
    public:
      std::vector<float> act(std::unordered_map<std::string, std::vector<float>>) override
      {
        return {0.0f};
      }
      void reset() override { reset_count += 1; }
      int reset_count = 0;
    };

    }  // namespace isaaclab

    int main() {
      const auto cfg = YAML::Load(R"yaml(
step_dt: 0.02
joint_ids_map: [0]
default_joint_pos: [0.0]
stiffness: [0.0]
damping: [0.0]
actions:
  ResetTestAction: {}
observations:
  actor_obs:
    reset_test_obs:
      params: {}
      history_length: 1
      scale: ~
      clip: ~
)yaml");
      auto robot = std::make_shared<isaaclab::Articulation>();
      isaaclab::ManagerBasedRLEnv env(cfg, robot);
      auto* alg = new isaaclab::ResetTestAlg();
      env.alg.reset(alg);
      robot->data.foot_event_summary.assign(80, 1.0f);
      robot->data.stair_latent_cache_valid = true;
      robot->data.prev_toe_vel_valid = true;
      bool callback_called = false;
      env.reset_callback = [&]() {
        callback_called = true;
        robot->data.foot_event_summary.assign(80, 0.0f);
      };
      env.reset();
      const bool summary_zero = std::all_of(
          robot->data.foot_event_summary.begin(),
          robot->data.foot_event_summary.end(),
          [](float value) { return value == 0.0f; });
      std::cout
          << (callback_called ? 1 : 0) << ' '
          << alg->reset_count << ' '
          << (summary_zero ? 1 : 0) << ' '
          << (robot->data.stair_latent_cache_valid ? 1 : 0) << ' '
          << (robot->data.prev_toe_vel_valid ? 1 : 0) << '\n';
      return 0;
    }
  """
  with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    source_path = tmp_path / "manager_reset_fixture.cpp"
    binary_path = tmp_path / "manager_reset_fixture"
    source_path.write_text(textwrap.dedent(source))
    ort_dir = ROOT / "thirdparty/onnxruntime-linux-x64-1.22.0"
    ort_lib = ort_dir / "lib/libonnxruntime.so.1.22.0"
    subprocess.run(
      [
        "g++",
        "-std=c++17",
        "-O2",
        "-I",
        str(ROOT / "include"),
        "-I",
        str(ort_dir / "include"),
        "-I",
        "/usr/include/eigen3",
        str(source_path),
        str(ort_lib),
        "-Wl,-rpath," + str(ort_dir / "lib"),
        "-lyaml-cpp",
        "-lfmt",
        "-o",
        str(binary_path),
      ],
      check=True,
    )
    output = subprocess.run(
      [str(binary_path)],
      check=True,
      text=True,
      capture_output=True,
    ).stdout
  return np.fromstring(output, sep=" ", dtype=np.float32)


def tensor_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
  return tuple(
    dimension.dim_value for dimension in value_info.type.tensor_type.shape.dim
  )


def main() -> None:
  deploy_text = DEPLOY_PATH.read_text()
  deploy_cfg = yaml.safe_load(deploy_text)
  assert "use_gym_history: true" not in deploy_text, (
    "mjlab actor history is term-major; use_gym_history=true makes it time-major"
  )
  command_cfg = deploy_cfg["commands"]["base_velocity"]
  assert command_cfg["ranges"]["lin_vel_x"] == [0.0, 1.0]
  assert command_cfg["ranges"]["lin_vel_y"] == [0.0, 0.0]
  assert command_cfg["ranges"]["ang_vel_z"] == [-0.8, 0.8]
  assert command_cfg["deadzone"] == 0.05

  latent_cfg = deploy_cfg["observations"]["latent_obs"]
  assert list(latent_cfg) == ["stair_latent", "foot_event_memory"]
  assert latent_cfg["stair_latent"]["params"]["include_gait_phase"] is True
  assert latent_cfg["foot_event_memory"]["params"]["summary_dim"] == 80
  observer_cfg = deploy_cfg["diagnostics"]["foot_event_observer"]
  assert observer_cfg["history_frames"] == 24
  footprint_detector = observer_cfg["footprint_touchdown_detector"]
  toe_detector = observer_cfg["toe_riser_detector"]
  assert footprint_detector["name"] == "footprint_touchdown_detector"
  assert toe_detector["name"] == "toe_riser_detector"
  assert footprint_detector["left_contact_index"] == 0
  assert footprint_detector["right_contact_index"] == 1
  assert footprint_detector["left_touchdown_index"] == 2
  assert footprint_detector["right_touchdown_index"] == 3
  assert toe_detector["left_toe_riser_index"] == 4
  assert toe_detector["right_toe_riser_index"] == 5
  assert footprint_detector["left_touchdown_threshold"] == 0.9661537409
  assert footprint_detector["right_touchdown_threshold"] == 0.9774025083
  assert toe_detector["left_toe_riser_threshold"] == 0.9886512756
  assert toe_detector["right_toe_riser_threshold"] == 0.9886512756
  assert not Path(footprint_detector["model_path"]).is_absolute()
  assert not Path(toe_detector["model_path"]).is_absolute()
  footprint_detector_path = resolve_repo_path(footprint_detector["model_path"])
  toe_detector_path = resolve_repo_path(toe_detector["model_path"])
  detector_history = np.zeros((1, 24, 93), dtype=np.float32)
  for detector_path in (footprint_detector_path, toe_detector_path):
    assert detector_path.exists()
    detector_model = onnx.load(detector_path)
    detector_inputs = {value.name: value for value in detector_model.graph.input}
    detector_outputs = {value.name: value for value in detector_model.graph.output}
    assert "obs_history" in detector_inputs
    assert "event_logits" in detector_outputs
    input_dims = detector_inputs["obs_history"].type.tensor_type.shape.dim
    output_dims = detector_outputs["event_logits"].type.tensor_type.shape.dim
    assert input_dims[1].dim_value == 24
    assert input_dims[2].dim_value == 93
    assert output_dims[1].dim_value == 6
    detector_session = ort.InferenceSession(
      str(detector_path),
      providers=["CPUExecutionProvider"],
    )
    detector_output = detector_session.run(
      ["event_logits"],
      {"obs_history": detector_history},
    )[0]
    assert detector_output.shape == (1, 6)
    assert np.isfinite(detector_output).all()
  memory_cfg = deploy_cfg["diagnostics"]["foot_event_memory"]
  assert memory_cfg["memory_len"] == 6
  assert memory_cfg["age_norm_s"] == 1.5
  assert memory_cfg["predicted_fill_enabled"] is True
  assert memory_cfg["predicted_fill_grace_frames"] == 3
  assert memory_cfg["predicted_fill_phase_window"] == 0.18
  assert memory_cfg["predicted_fill_phase_lead_window"] == 0.08
  assert memory_cfg["predicted_fill_contact_threshold"] == 0.35
  assert memory_cfg["predicted_fill_confidence"] == 0.35
  assert memory_cfg["predicted_fill_touchdown_prob"] == 0.35
  assert memory_cfg["touchdown_gate_enabled"] is True
  assert memory_cfg["touchdown_gate_contact_threshold"] == 0.20
  assert memory_cfg["ratchet_probe_increment_m"] == 0.05
  assert memory_cfg["ratchet_probe_bootstrap_increment_m"] == 0.10
  assert memory_cfg["ratchet_probe_cap_m"] == 0.76
  assert memory_cfg["ratchet_first_collision_enters_stair_mode"] is True
  assert memory_cfg["ratchet_single_collision_confirms_interval"] is False
  assert memory_cfg["ratchet_min_stride_m"] == 0.10
  assert memory_cfg["ratchet_max_stride_m"] == 0.85

  model = onnx.load(MODEL_PATH)
  inputs = {value.name: tensor_shape(value) for value in model.graph.input}
  outputs = {value.name: tensor_shape(value) for value in model.graph.output}

  assert inputs == {
    "actor_obs": (1, 490),
    "latent_obs": (1, 173),
    "h_in": (1, 1, 128),
    "c_in": (1, 1, 128),
    "z_in": (1, 24),
    "gate_state_in": (1, 5),
  }
  assert outputs["actions"] == (1, 29)
  assert outputs["h_out"] == inputs["h_in"]
  assert outputs["c_out"] == inputs["c_in"]
  assert outputs["z_out"] == inputs["z_in"]
  assert outputs["gate_state_out"] == inputs["gate_state_in"]
  assert outputs["safe_stride_interval"] == (1, 2)
  assert outputs["safe_stride_confidence"] == (1, 1)

  metadata = {item.key: item.value for item in model.metadata_props}
  assert metadata["policy_latent_obs_dim"] == "173"
  assert metadata["policy_slow_latent_dim"] == "24"
  assert metadata["policy_stair_safe_stride_min"] == "0.1"
  assert metadata["policy_stair_safe_stride_max"] == "0.85"
  expected_default_pos = np.fromstring(metadata["default_joint_pos"], sep=",")
  hip_stiffness = 40.17923863450712
  hip_roll_stiffness = 99.09842777666111
  linkage_stiffness = 28.50124619574858
  arm_stiffness = 14.25062309787429
  wrist_stiffness = 16.77832748089279
  expected_stiffness = np.array(
    [
      hip_stiffness, hip_roll_stiffness, hip_stiffness,
      hip_roll_stiffness, linkage_stiffness, linkage_stiffness,
      hip_stiffness, hip_roll_stiffness, hip_stiffness,
      hip_roll_stiffness, linkage_stiffness, linkage_stiffness,
      hip_stiffness, linkage_stiffness, linkage_stiffness,
      arm_stiffness, arm_stiffness, arm_stiffness, arm_stiffness, arm_stiffness,
      wrist_stiffness, wrist_stiffness,
      arm_stiffness, arm_stiffness, arm_stiffness, arm_stiffness, arm_stiffness,
      wrist_stiffness, wrist_stiffness,
    ]
  )
  hip_damping = 2.557889775413375
  hip_roll_damping = 6.308801853496639
  linkage_damping = 1.814445686584846
  arm_damping = 0.907222843292423
  wrist_damping = 1.06814150219
  expected_damping = np.array(
    [
      hip_damping, hip_roll_damping, hip_damping,
      hip_roll_damping, linkage_damping, linkage_damping,
      hip_damping, hip_roll_damping, hip_damping,
      hip_roll_damping, linkage_damping, linkage_damping,
      hip_damping, linkage_damping, linkage_damping,
      arm_damping, arm_damping, arm_damping, arm_damping, arm_damping,
      wrist_damping, wrist_damping,
      arm_damping, arm_damping, arm_damping, arm_damping, arm_damping,
      wrist_damping, wrist_damping,
    ]
  )
  expected_action_scale = np.array(
    [
      0.547546462991107,
      0.350661466378824,
      0.547546462991107,
      0.350661466378824,
      0.438577313923367,
      0.438577313923367,
      0.547546462991107,
      0.350661466378824,
      0.547546462991107,
      0.350661466378824,
      0.438577313923367,
      0.438577313923367,
      0.547546462991107,
      0.438577313923367,
      0.438577313923367,
      0.438577313923367,
      0.438577313923367,
      0.438577313923367,
      0.438577313923367,
      0.438577313923367,
      0.0745008703295071,
      0.0745008703295071,
      0.438577313923367,
      0.438577313923367,
      0.438577313923367,
      0.438577313923367,
      0.438577313923367,
      0.0745008703295071,
      0.0745008703295071,
    ]
  )
  np.testing.assert_allclose(deploy_cfg["stiffness"], expected_stiffness, atol=1e-12)
  np.testing.assert_allclose(deploy_cfg["damping"], expected_damping, atol=1e-12)
  np.testing.assert_allclose(
    deploy_cfg["default_joint_pos"], expected_default_pos, atol=1e-12
  )
  np.testing.assert_allclose(
    deploy_cfg["actions"]["JointPositionAction"]["scale"],
    expected_action_scale,
    atol=1e-15,
  )
  np.testing.assert_allclose(
    deploy_cfg["actions"]["JointPositionAction"]["offset"],
    expected_default_pos,
    atol=1e-12,
  )

  initializers = {
    initializer.name: numpy_helper.to_array(initializer)
    for initializer in model.graph.initializer
  }
  actor_mean = initializers["actor_obs_normalizer._mean"].reshape(-1)
  assert actor_mean.shape == (490,)

  # Term-major layout:
  # base_ang_vel 3*5 occupies [0:15], projected_gravity 3*5 [15:30].
  gravity_history = actor_mean[15:30].reshape(5, 3)
  assert np.all(gravity_history[:, 2] < -0.9)
  assert np.all(np.abs(gravity_history[:, :2]) < 0.1)

  stair_latent = np.zeros((1, 93), dtype=np.float32)
  foot_summary = np.zeros((1, 80), dtype=np.float32)
  latent_obs = np.concatenate([stair_latent, foot_summary], axis=1)
  assert latent_obs.shape == inputs["latent_obs"]
  np.testing.assert_allclose(latent_obs[:, 93:], 0.0)

  session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
  ort_inputs = {
    "actor_obs": np.zeros(inputs["actor_obs"], dtype=np.float32),
    "latent_obs": latent_obs,
    "h_in": np.zeros(inputs["h_in"], dtype=np.float32),
    "c_in": np.zeros(inputs["c_in"], dtype=np.float32),
    "z_in": np.zeros(inputs["z_in"], dtype=np.float32),
    "gate_state_in": np.zeros(inputs["gate_state_in"], dtype=np.float32),
  }
  ort_outputs = dict(zip([o.name for o in session.get_outputs()], session.run(None, ort_inputs)))
  assert ort_outputs["actions"].shape == outputs["actions"]
  assert np.isfinite(ort_outputs["actions"]).all()
  assert ort_outputs["z_out"].shape == inputs["z_in"]
  assert ort_outputs["safe_stride_interval"].shape == (1, 2)
  assert ort_outputs["safe_stride_confidence"].shape == (1, 1)
  manager_reset = run_cpp_manager_reset_fixture()
  np.testing.assert_allclose(
    manager_reset,
    np.array([1, 1, 1, 0, 0], dtype=np.float32),
    atol=0.0,
  )

  observer_reset = run_cpp_observer_reset_fixture(
    footprint_detector_path,
    toe_detector_path,
  )
  assert observer_reset[0] == 2.0
  assert observer_reset[1] > 0.0
  assert observer_reset[2] == 1.0
  assert observer_reset[3] == 0.0

  fail_safe_action_size, fail_safe_all_zero, recurrent_before, recurrent_after = (
    run_cpp_ort_fail_safe_fixture()
  )
  assert fail_safe_action_size == 29
  assert fail_safe_all_zero
  assert recurrent_before > 0
  assert recurrent_after == 0

  odom_fixture = run_cpp_foot_odometry_fixture()
  np.testing.assert_allclose(
    odom_fixture,
    np.array([0.30, 0.50, 0.30, 0.30, 0.30, 0.50, 0.15], dtype=np.float32),
    atol=2.5e-5,
  )

  predicted_fill = run_cpp_predicted_fill_fixture()
  np.testing.assert_allclose(
    predicted_fill,
    np.array(
      [
        0, 1, 0, 1, 0.35, 0.35, 0, 1, 1, 1, 0, 0, 0, 1,
        0, 1, 1, 1, 0, 0.20, 0.25, 1, 0,
      ],
      dtype=np.float32,
    ),
    atol=2.5e-5,
  )

  cpp_fixture = run_cpp_foot_event_memory_fixture()
  python_fixture = run_python_foot_event_memory_fixture(memory_cfg)
  assert cpp_fixture.keys() == python_fixture.keys()
  for label in sorted(cpp_fixture):
    assert cpp_fixture[label].shape == (80,)
    diff = np.abs(cpp_fixture[label] - python_fixture[label])
    if np.any(diff > 2.5e-5):
      bad = np.flatnonzero(diff > 2.5e-5)
      raise AssertionError(
        f"C++ FootEventMemory parity failed for {label}: "
        f"indices={bad.tolist()} "
        f"cpp={cpp_fixture[label][bad].tolist()} "
        f"python={python_fixture[label][bad].tolist()}"
      )
    np.testing.assert_allclose(
      cpp_fixture[label],
      python_fixture[label],
      atol=2.5e-5,
      rtol=2.5e-5,
      err_msg=f"C++ FootEventMemory parity failed for {label}",
    )

  np.testing.assert_allclose(cpp_fixture["zero"], 0.0, atol=1.0e-7)
  np.testing.assert_allclose(cpp_fixture["after_reset"], 0.0, atol=1.0e-7)
  assert cpp_fixture["probe_first_hit"][70] == 1.0
  assert cpp_fixture["probe_first_hit"][76] == 0.0
  assert cpp_fixture["probe_bootstrap"][78] == 1.0
  assert cpp_fixture["probe_small_1"][79] == 0.0
  assert cpp_fixture["probe_cap_not_reached"][72] <= 0.76001
  assert cpp_fixture["probe_cap_not_reached"][79] == 0.0
  assert cpp_fixture["lock_second_hit"][76] == 1.0
  assert cpp_fixture["lock_second_hit"][79] == 0.5
  assert cpp_fixture["lock_recovery"][76] == 1.0
  assert cpp_fixture["lock_recovery"][79] == 1.0, cpp_fixture["lock_recovery"][70:80]
  assert cpp_fixture["lock_reopen"][76] == 1.0
  assert cpp_fixture["lock_reopen"][79] == 0.5
  assert cpp_fixture["predicted_pair"][0] == 1.0
  assert 0.18 < cpp_fixture["predicted_pair"][1] < 0.20

  print("Slow-latent ONNX/deploy contract test passed.")


def test_slow_latent_contract() -> None:
  main()


if __name__ == "__main__":
  main()
