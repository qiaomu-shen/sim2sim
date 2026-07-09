# G1 Sim2Sim 独立地形环境

这个目录只负责创建和选择仿真测试地形，不修改原有
`unitree_robots/g1/scene.xml`，也不修改部署策略。

生成的场景位于 `unitree_robots/g1/scene_sim2sim_*.xml`。它们与
`g1_29dof.xml` 放在同一目录，因此继续使用原来的机器人模型、关节、
传感器和网格资源，场景之间只有地形不同。

## 查看可用地形

```bash
cd unitree_mujoco
python3 sim2sim_terrains/terrain_env.py list
```

初始地形包括：

- `flat`：纯平地基准；
- `stairs_08`、`stairs_12`、`stairs_15`、`stairs_20`：不同级高的连续上下楼梯；
- `stepping_stones`：可复现的梅花桩路线；
- `rough`：固定随机种子的离散粗糙地形；
- `gaps`：沟宽逐渐增加的跨沟路线；
- `mixed`：粗糙、梅花桩、楼梯串联路线。

## 生成和启动

只生成一个场景：

```bash
python3 sim2sim_terrains/terrain_env.py build stairs_15
```

生成全部场景：

```bash
python3 sim2sim_terrains/terrain_env.py build-all
```

直接选择并启动：

```bash
python3 sim2sim_terrains/terrain_env.py run stairs_15
```

等价的底层命令是：

```bash
simulate/build/unitree_mujoco \
  --scene unitree_robots/g1/scene_sim2sim_stairs_15.xml
```

随后控制器仍按原方式单独启动：

```bash
../deploy_lstm/robots/g1/build/g1_ctrl --network sim
```

## 新增地形

在 `terrain_env.py` 中：

1. 向 `TERRAINS` 增加一个名称；
2. 在 `_populate()` 为这个名称组合已有生成函数；
3. 或新增一个 `add_*()` 函数，再通过 `build`/`run` 使用。

所有随机地形都必须使用固定 seed，保证不同策略在完全相同的场景上比较。
MuJoCo 的 `box size` 是半尺寸；本工具的 `full_size` 和楼梯参数均使用实际全尺寸，
避免手工编辑 XML 时把尺寸放大或缩小两倍。
