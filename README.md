# Sim2Sim Unitree G1 Workspace

这是 `test/lidar-blindzone-heightmap` 分支，用于 Unitree G1 29DOF sim2sim 部署、激光雷达高度图桥接和实机/仿真一致性调试。

## 分支目标

本分支围绕 `unitree_rl_lab/deploy/robots/g1_29dof` 的 G1-29DOF 控制器展开，重点是：

- 将原 21DOF active policy 迁移到 29DOF 机体，并锁住新增的 8 个自由度。
- 保持 policy 的 21 个 action/observation 槽位顺序，不按 29DOF SDK natural order 重新编号。
- 接入 Livox 点云到 11x11 `height_scan` 的高度图桥接。
- 为平地、楼梯和跌倒 case 增加 MuJoCo/部署侧诊断输出。

## 主要目录

- `unitree_rl_lab/deploy/robots/g1_29dof/`：G1-29DOF 控制器、策略配置、heightmap bridge 和诊断代码。
- `unitree_mujoco/`：Unitree MuJoCo 仿真环境、机器人 XML、场景和 DDS bridge。
- `deploy_lstm/`：早期 LSTM/感知部署实验和辅助分析脚本。
- `unitree_sdk2/`：Unitree SDK2 依赖与示例。

## 当前策略

当前调试策略位于：

```text
unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/lidar_blindwalking_model2600
```

该策略的 ONNX 输入输出约定：

- observation: `[1, 193]`
- action: `[1, 21]`
- `height_scan`: 121 维

## 本分支运行顺序

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

## 注意

这是机器人部署调试工作区。实机测试前需要确认急停、吊挂/保护措施、网络 domain、DDS topic 和策略/关节顺序完全匹配。
