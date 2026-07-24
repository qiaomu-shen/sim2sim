# Sim2Sim Workspace

This is the `main` branch for the sim2sim workspace. It is kept as the
repository entry point and branch index for Unitree G1/G1-29DOF simulation,
deployment, and policy transfer experiments.

## Branches

- `main`: base workspace entry point. Use this branch to find shared
  dependencies, directory layout, and the purpose of each working branch.
- `test/lidar-blindzone-heightmap`: G1-29DOF lidar blindwalking / heightmap
  bridge deployment branch. It records the current 21DOF active-policy to
  29DOF locked-joint migration work, lidar height-scan bridge scripts, and
  MuJoCo / real-robot deployment diagnostics.

## Repository Layout

- `unitree_rl_lab/`: training and deployment code, including G1-29DOF policy
  deployment.
- `unitree_mujoco/`: Unitree MuJoCo simulation and sim2sim support.
- `deploy_lstm/`: deployment and analysis workspace used by locomotion tests.
- `unitree_sdk2/`: Unitree SDK2 dependency and examples.
- `tools/`: local helper tools.

## Usage

Check the README on the branch you are working on before running an experiment.
Robot deployment should always confirm emergency stop, network interface,
DDS domain, joint order, policy observation order, and locked-joint settings
before enabling torque.
