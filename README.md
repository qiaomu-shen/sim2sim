# Sim2Sim Unitree G1 工作区 / Workspace

语言 / Language: [中文](#中文) | [English](#english)

<a id="中文"></a>
## 中文

这是 `test/lidar-blindzone-heightmap` 分支，用于 Unitree G1 29DOF
sim2sim 部署、激光雷达高度图桥接和实机/仿真一致性调试。

### 分支目标

本分支围绕 `unitree_rl_lab/deploy/robots/g1_29dof` 的 G1-29DOF 控制器展开，重点是：

- 将原 21DOF active policy 迁移到 29DOF 机体，并锁住新增的 8 个自由度。
- 保持 policy 的 21 个 action/observation 槽位顺序，不按 29DOF SDK natural order 重新编号。
- 接入 Livox 点云到 11x11 `height_scan` 的高度图桥接。
- 为平地、楼梯和跌倒 case 增加 MuJoCo/部署侧诊断输出。

### Sim2Sim 演示视频

![G1 lidar heightmap sim2sim stable walk](docs/media/lidar_sim2sim_stable_walk.gif)

高清 MP4：`docs/media/lidar_sim2sim_stable_walk.mp4`

这段是本分支的激光雷达 `height_scan` sim2sim 演示：G1-29DOF 在 MuJoCo
台阶场景中使用 lidar heightmap 输入稳定跨越石块序列。原始录屏为
`simplescreenrecorder-2026-07-17_14.14.42.mp4`，已检查并只保留有效段
`7.2s-16.25s`；`7s` 前主要是起步/等待，`16.3s` 后开始进入终点平台和不稳姿态，未放入 README 演示视频。

### 主要目录

- `unitree_rl_lab/deploy/robots/g1_29dof/`：G1-29DOF 控制器、策略配置、heightmap bridge 和诊断代码。
- `unitree_mujoco/`：Unitree MuJoCo 仿真环境、机器人 XML、场景和 DDS bridge。
- `deploy_lstm/`：早期 LSTM/感知部署实验和辅助分析脚本。
- `unitree_sdk2/`：Unitree SDK2 依赖与示例。

### 当前策略

当前调试策略位于：

```text
unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/lidar_blindwalking_model2600
```

该策略的 ONNX 输入输出约定：

- observation: `[1, 193]`
- action: `[1, 21]`
- `height_scan`: 121 维

### 本分支运行顺序

终端 1，启动 MuJoCo：

```bash
cd /home/ubt2204/work/111/TRY/sim2sim/unitree_mujoco/simulate
./build/unitree_mujoco --network lo --domain_id 0
```

终端 2，启动高度图桥接：

```bash
cd /home/ubt2204/work/111/TRY/sim2sim/unitree_rl_lab/deploy/robots/g1_29dof
./scripts/run_lidar_blindwalking_model2600_heightmap_bridge.sh
```

终端 3，启动控制器：

```bash
cd /home/ubt2204/work/111/TRY/sim2sim/unitree_rl_lab/deploy/robots/g1_29dof
./build/g1_ctrl --network lo
```

### 注意

这是机器人部署调试工作区。实机测试前需要确认急停、吊挂/保护措施、网络
domain、DDS topic 和策略/关节顺序完全匹配。

### 许可证

本工作区原创代码、文档、脚本和演示媒体默认使用 MIT License，见
`LICENSE`。第三方组件保留各自许可证，见 `THIRD_PARTY_NOTICES.md` 和各
third-party 目录内的 license 文件。

<a id="english"></a>
## English

This is the `test/lidar-blindzone-heightmap` branch for Unitree G1 29DOF
sim2sim deployment, lidar heightmap bridging, and real/simulation consistency
debugging.

### Branch Goals

This branch focuses on the G1-29DOF controller under
`unitree_rl_lab/deploy/robots/g1_29dof`:

- Migrate the original 21DOF active policy to the 29DOF body while locking the
  8 additional joints.
- Keep the policy's 21 action/observation slots in the original order instead
  of remapping them to the 29DOF SDK natural order.
- Bridge Livox point clouds into an 11x11 `height_scan`.
- Add MuJoCo and deployment-side diagnostics for flat-ground, stair, and fall
  cases.

### Sim2Sim Demo

![G1 lidar heightmap sim2sim stable walk](docs/media/lidar_sim2sim_stable_walk.gif)

HD MP4: `docs/media/lidar_sim2sim_stable_walk.mp4`

This lidar `height_scan` sim2sim demo shows the G1-29DOF robot stably crossing
the stepping-stone sequence in the MuJoCo stair scene using lidar heightmap
input. The source recording is `simplescreenrecorder-2026-07-17_14.14.42.mp4`.
After inspection, only `7.2s-16.25s` is kept. Frames before `7s` are mostly
startup/waiting, and frames after `16.3s` start to enter the end platform and
unstable finish posture, so they are excluded from the README demo.

### Main Directories

- `unitree_rl_lab/deploy/robots/g1_29dof/`: G1-29DOF controller, policy config,
  heightmap bridge, and diagnostics.
- `unitree_mujoco/`: Unitree MuJoCo simulation, robot XML files, scenes, and
  DDS bridge.
- `deploy_lstm/`: earlier LSTM/perception deployment experiments and analysis
  helpers.
- `unitree_sdk2/`: Unitree SDK2 dependency and examples.

### Current Policy

The current policy under test is:

```text
unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/lidar_blindwalking_model2600
```

ONNX input/output convention:

- observation: `[1, 193]`
- action: `[1, 21]`
- `height_scan`: 121 dimensions

### Run Order

Terminal 1, start MuJoCo:

```bash
cd /home/ubt2204/work/111/TRY/sim2sim/unitree_mujoco/simulate
./build/unitree_mujoco --network lo --domain_id 0
```

Terminal 2, start the heightmap bridge:

```bash
cd /home/ubt2204/work/111/TRY/sim2sim/unitree_rl_lab/deploy/robots/g1_29dof
./scripts/run_lidar_blindwalking_model2600_heightmap_bridge.sh
```

Terminal 3, start the controller:

```bash
cd /home/ubt2204/work/111/TRY/sim2sim/unitree_rl_lab/deploy/robots/g1_29dof
./build/g1_ctrl --network lo
```

### Safety

This is a robot deployment debugging workspace. Before real-robot tests,
confirm emergency stop, suspension/protection, network domain, DDS topics,
and exact policy/joint ordering.

### License

Original workspace code, documentation, scripts, and included demo media are
licensed under the MIT License unless otherwise noted. See `LICENSE`.
Vendored third-party components keep their own licenses; see
`THIRD_PARTY_NOTICES.md` and the license files in each third-party directory.
