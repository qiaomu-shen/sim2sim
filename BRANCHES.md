# 分支说明 / Branch Notes

语言 / Language: [中文](#中文) | [English](#english)

<a id="中文"></a>
## 中文

这个文件记录当前仓库各分支的用途。

| 分支 | 用途 |
| --- | --- |
| `main` | 仓库入口、公共目录说明、许可证说明和分支索引。 |
| `test/lidar-blindzone-heightmap` | G1-29DOF lidar blindwalking 部署分支，包含 heightmap bridge、sim2sim 演示、实机/仿真诊断和 21DOF policy 迁移说明。 |

### 当前备注

- 分支专属运行命令放在该分支自己的 `README.md`。
- 机器人相关安全检查应靠近部署脚本和策略配置文件维护。
- 实验性机器人控制改动合回 `main` 前，需要确认实机/仿真的 observation 顺序和关节映射已经对齐。

<a id="english"></a>
## English

This file records the intended role of each branch in the repository.

| Branch | Purpose |
| --- | --- |
| `main` | Repository entry point, shared layout notes, licensing information, and branch index. |
| `test/lidar-blindzone-heightmap` | G1-29DOF lidar blindwalking deployment branch with heightmap bridge, sim2sim demo media, real/simulation diagnostics, and 21DOF policy migration notes. |

### Current Notes

- Keep branch-specific run commands in that branch's `README.md`.
- Keep robot-specific safety checks close to deployment scripts and policy
  config files.
- Do not merge experimental robot-control changes back to `main` until the
  real/simulation observation order and joint mapping have been verified.
