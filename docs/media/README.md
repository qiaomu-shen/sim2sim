# Sim2Sim Demo / 演示

This page documents the lidar-heightmap sim2sim demo used in the repository READMEs.

本页面用于统一展示仓库 README 中使用的 lidar-heightmap sim2sim 演示。

![G1 lidar heightmap sim2sim stable walk](https://raw.githubusercontent.com/qiaomu-shen/sim2sim/test/lidar-blindzone-heightmap/docs/media/lidar_sim2sim_stable_walk.gif)

## HD Video / 高清视频

[lidar_sim2sim_stable_walk.mp4](https://github.com/qiaomu-shen/sim2sim/blob/test/lidar-blindzone-heightmap/docs/media/lidar_sim2sim_stable_walk.mp4)

## Demo Content / 演示内容

- **Robot:** Unitree G1-29DOF
- **Perception:** lidar-derived 11 × 11 `height_scan`
- **Scene:** MuJoCo stepping-stone / stair scene
- **Experiment branch:** `test/lidar-blindzone-heightmap`
- **Preview:** GIF generated from the MP4 clip

The robot uses the lidar heightmap input to stably cross the stepping-stone sequence in simulation.

机器人使用 lidar 高度图输入，在 MuJoCo 仿真中稳定跨越石块序列。

## Media Files / 媒体文件

The binary demo assets are maintained on the experiment branch:

- `docs/media/lidar_sim2sim_stable_walk.gif`
- `docs/media/lidar_sim2sim_stable_walk.mp4`
- `docs/media/lidar_sim2sim_stable_walk_poster.jpg`

二进制演示文件统一维护在 `test/lidar-blindzone-heightmap` 分支，两个分支的 README 与本页面均引用同一份演示资源，避免内容不一致。
