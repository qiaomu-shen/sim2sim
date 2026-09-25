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


## Current Branch: `test/lidar-blindzone-heightmap`

This branch focuses on **G1-29DOF sim2sim deployment with lidar heightmap input** and on reducing the gap between the training/simulation interface and the deployable robot runtime.

本分支重点研究 **G1-29DOF 的激光雷达高度图 sim2sim 部署**，并处理训练/仿真接口与机器人实际运行接口之间的一致性问题。

### Main Goals / 主要目标

- Migrate the original **21DOF active policy** to the **29DOF G1 body** while locking the 8 additional joints.  
  将原 **21DOF active policy** 迁移到 **29DOF G1 机体**，并锁定新增的 8 个自由度。
- Preserve the original 21 policy action/observation slots instead of re-indexing them according to the 29DOF SDK natural order.  
  保持原策略的 21 个 action / observation 槽位顺序，不按照 29DOF SDK natural order 重新编号。
- Bridge **Livox point clouds** into an **11 × 11 `height_scan`** representation.  
  将 **Livox 点云**转换为 **11 × 11 `height_scan`**。
- Add MuJoCo- and deployment-side diagnostics for flat-ground, stair, and fall cases.  
  为平地、楼梯和跌倒等情况增加 MuJoCo 与部署侧诊断。

## Sim2Sim Demo / 演示

![G1 lidar heightmap sim2sim stable walk](docs/media/lidar_sim2sim_stable_walk.gif)

HD video / 高清视频：`docs/media/lidar_sim2sim_stable_walk.mp4`

The demo shows the G1-29DOF robot using lidar-derived `height_scan` input to stably cross the stepping-stone sequence in the MuJoCo stair scene.

该演示展示 G1-29DOF 在 MuJoCo 台阶场景中使用 lidar 生成的 `height_scan` 输入稳定跨越石块序列。

## Current Policy / 当前策略

The policy currently used for this deployment experiment is located at:

~~~text
unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/lidar_blindwalking_model2600
~~~

ONNX interface / ONNX 输入输出：

| Item | Shape |
| --- | ---: |
| Observation | `[1, 193]` |
| Action | `[1, 21]` |
| `height_scan` | 121 dims |

The main controller and bridge code are under:

~~~text
unitree_rl_lab/deploy/robots/g1_29dof/
~~~

## Sim2Sim Run Order / 运行顺序

Run the following commands from the repository root.

从仓库根目录开始，按以下顺序运行。

### Terminal 1 — MuJoCo

~~~bash
cd unitree_mujoco/simulate
./build/unitree_mujoco --network lo --domain_id 0
~~~

### Terminal 2 — Heightmap bridge

~~~bash
cd unitree_rl_lab/deploy/robots/g1_29dof
./scripts/run_lidar_blindwalking_model2600_heightmap_bridge.sh
~~~

### Terminal 3 — G1 controller

~~~bash
cd unitree_rl_lab/deploy/robots/g1_29dof
./build/g1_ctrl --network lo
~~~

## Deployment Notes / 部署说明

This branch is an experimental robot-deployment workspace. Before real-robot testing, verify:

- emergency stop, suspension, and physical protection;
- network interface, DDS domain, and DDS topics;
- exact policy observation/action ordering;
- active-joint and locked-joint mapping between the 21DOF policy and the 29DOF robot;
- consistency between the heightmap bridge and the policy input convention.

本分支属于机器人部署调试工作区。实机测试前请确认：

- 急停、吊挂与物理保护措施；
- 网络接口、DDS domain 与 DDS topic；
- policy observation / action 顺序完全一致；
- 21DOF policy 与 29DOF 机器人之间的 active / locked joint 映射；
- heightmap bridge 与策略输入约定一致。

## License / 许可证

Original code, documentation, scripts, and included demo media in this workspace are licensed under the MIT License unless otherwise noted. See `LICENSE`.

本工作区原创代码、文档、脚本和演示媒体默认使用 MIT License，详见 `LICENSE`。

Vendored third-party components retain their own licenses; see `THIRD_PARTY_NOTICES.md` and the license files inside the corresponding third-party directories.
