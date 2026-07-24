# Sim2Sim 工作区 / Workspace

语言 / Language: [中文](#中文) | [English](#english)

<a id="中文"></a>
## 中文

这是 `main` 分支，作为 sim2sim 工作区的入口和分支索引。仓库包含 Unitree
G1/G1-29DOF 仿真、部署和策略迁移实验相关代码。

### 分支说明

- `main`：基础工作区入口。这里保留公共依赖、目录索引、许可证说明和其他分支的简要介绍。
- `test/lidar-blindzone-heightmap`：G1-29DOF lidar blindwalking / heightmap
  bridge 部署分支。该分支记录 21DOF active policy 迁移到 29DOF 锁关节机体、
  lidar `height_scan` 桥接脚本、MuJoCo sim2sim 演示和实机/仿真诊断。

### 主要目录

- `unitree_rl_lab/`：训练与部署代码，包括 G1-29DOF 策略部署。
- `unitree_mujoco/`：Unitree MuJoCo 仿真环境和 sim2sim 支持。
- `deploy_lstm/`：locomotion 测试中使用的部署与分析工作区。
- `unitree_sdk2/`：Unitree SDK2 依赖和示例。
- `tools/`：本地辅助工具。

### 使用建议

具体实验和运行步骤请看对应分支的 README。实机部署前请确认急停、网络接口、
DDS domain、DDS topic、关节顺序、策略观测顺序和锁关节设置。

### 许可证

本工作区原创代码、文档、脚本和演示媒体默认使用 MIT License，见
`LICENSE`。第三方组件保留各自许可证，见 `THIRD_PARTY_NOTICES.md` 和各
third-party 目录内的 license 文件。

<a id="english"></a>
## English

This is the `main` branch, used as the entry point and branch index for the
sim2sim workspace. The repository contains code for Unitree G1/G1-29DOF
simulation, deployment, and policy-transfer experiments.

### Branches

- `main`: base workspace entry point. This branch keeps shared dependencies,
  repository layout notes, licensing information, and a short introduction to
  the other branches.
- `test/lidar-blindzone-heightmap`: G1-29DOF lidar blindwalking / heightmap
  bridge deployment branch. It records the 21DOF active-policy migration to a
  29DOF locked-joint body, lidar `height_scan` bridge scripts, MuJoCo sim2sim
  demo media, and real/simulation diagnostics.

### Repository Layout

- `unitree_rl_lab/`: training and deployment code, including G1-29DOF policy
  deployment.
- `unitree_mujoco/`: Unitree MuJoCo simulation and sim2sim support.
- `deploy_lstm/`: deployment and analysis workspace used by locomotion tests.
- `unitree_sdk2/`: Unitree SDK2 dependency and examples.
- `tools/`: local helper tools.

### Usage

Check the README on the branch you are working on before running an experiment.
Before real-robot deployment, confirm emergency stop, network interface, DDS
domain, DDS topics, joint order, policy observation order, and locked-joint
settings.

### License

Original workspace code, documentation, scripts, and included demo media are
licensed under the MIT License unless otherwise noted. See `LICENSE`.
Vendored third-party components keep their own licenses; see
`THIRD_PARTY_NOTICES.md` and the license files in each third-party directory.
