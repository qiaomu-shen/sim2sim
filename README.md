# Unitree G1 Sim2Sim & Deployment Workspace / Unitree G1 仿真迁移与部署工作区

This repository is a working space for **Unitree G1 / G1-29DOF simulation, sim2sim policy transfer, perception bridging, and deployment debugging**. It collects the runtime, MuJoCo simulation, deployment code, and experiment-specific branches used to move locomotion policies from training/simulation toward deployable G1 systems.

本仓库用于 **Unitree G1 / G1-29DOF 的仿真、sim2sim 策略迁移、感知桥接与部署调试**，集中保存 MuJoCo 仿真、控制器、部署代码以及不同实验分支，目标是将训练得到的运动策略逐步迁移到可部署的 G1 系统中。

> **Main focus / 主要方向:** simulation → sim2sim → deployment → real/simulation consistency

## Branches / 分支

| Branch | Purpose / 用途 |
| --- | --- |
| **`main`** | Base workspace, shared dependencies, repository index, and common deployment infrastructure. / 基础工作区、公共依赖、仓库索引与通用部署基础设施。 |
| **`test/lidar-blindzone-heightmap`** | G1-29DOF lidar blindwalking / heightmap deployment branch, including 21DOF-policy migration, Livox heightmap bridging, MuJoCo demos, and deployment diagnostics. / G1-29DOF 激光雷达盲走与高度图部署分支，包括 21DOF 策略迁移、Livox 高度图桥接、MuJoCo 演示和部署诊断。 |

## Repository Layout / 代码结构

- `unitree_rl_lab/` — training/deployment code and G1-29DOF policy runtime. / 训练与部署代码，包括 G1-29DOF 策略运行。
- `unitree_mujoco/` — Unitree MuJoCo simulation, robot XMLs, scenes, and sim2sim support. / Unitree MuJoCo 仿真、机器人 XML、场景与 sim2sim 支持。
- `deploy_lstm/` — earlier locomotion, recurrent-policy, and perception deployment experiments. / 早期运动控制、循环策略与感知部署实验。
- `unitree_sdk2/` — Unitree SDK2 dependency and examples. / Unitree SDK2 依赖与示例。
- `tools/` — local helper and diagnostic tools. / 本地辅助与诊断工具。


## Current Branch: `main`

The `main` branch acts as the **entry point and shared workspace** for the repository. It keeps the common dependency layout, deployment infrastructure, licensing information, and an index of experiment branches.

`main` 分支作为整个仓库的**入口与公共工作区**，主要维护通用依赖、部署基础设施、许可证信息以及实验分支索引。

For the lidar-heightmap G1-29DOF deployment experiment, switch to:

~~~bash
git switch test/lidar-blindzone-heightmap
~~~

该分支包含更完整的 G1-29DOF lidar blindwalking / heightmap 实验说明、演示视频和运行步骤。

## Deployment Notes / 部署说明

This repository contains experimental robot-deployment code. Before running on hardware, verify:

- emergency stop and physical protection;
- network interface and DDS domain;
- DDS topics and communication status;
- robot joint ordering and active/locked-joint mapping;
- policy observation and action ordering.

本仓库包含机器人部署实验代码。实机运行前请确认：

- 急停与物理保护措施；
- 网络接口与 DDS domain；
- DDS topic 与通信状态；
- 机器人关节顺序以及 active / locked joint 映射；
- policy observation / action 顺序。

## License / 许可证

Original code, documentation, scripts, and included demo media in this workspace are licensed under the MIT License unless otherwise noted. See `LICENSE`.

本工作区原创代码、文档、脚本和演示媒体默认使用 MIT License，详见 `LICENSE`。

Vendored third-party components retain their own licenses; see `THIRD_PARTY_NOTICES.md` and the license files inside the corresponding third-party directories.
