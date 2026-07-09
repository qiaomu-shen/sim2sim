"""Static deployment-contract checks for the G1 slow-latent policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "robots/g1/config/policy/velocity/slow_latent"
MODEL_PATH = POLICY_DIR / "exported/policy.onnx"
DEPLOY_PATH = POLICY_DIR / "params/deploy.yaml"


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

  model = onnx.load(MODEL_PATH)
  inputs = {value.name: tensor_shape(value) for value in model.graph.input}
  outputs = {value.name: tensor_shape(value) for value in model.graph.output}

  assert inputs == {
    "actor_obs": (1, 490),
    "latent_obs": (1, 91),
    "h_in": (1, 1, 128),
    "c_in": (1, 1, 128),
    "z_in": (1, 16),
    "gate_state_in": (1, 5),
  }
  assert outputs["actions"] == (1, 29)
  assert outputs["h_out"] == inputs["h_in"]
  assert outputs["c_out"] == inputs["c_in"]
  assert outputs["z_out"] == inputs["z_in"]
  assert outputs["gate_state_out"] == inputs["gate_state_in"]

  metadata = {item.key: item.value for item in model.metadata_props}
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

  print("Slow-latent ONNX/deploy contract test passed.")


if __name__ == "__main__":
  main()
