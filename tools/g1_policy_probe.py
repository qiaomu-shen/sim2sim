#!/usr/bin/env python3
"""Headless MuJoCo probe for the G1 stepping-stone policy.

This bypasses DDS, joystick, and the C++ controller. It is meant to answer one
question: does the exported policy close the loop in MuJoCo with a given joint
ordering and low-level PD model?
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO / "unitree_mujoco/unitree_robots/g1/scene_sim2sim_flat.xml"
DEFAULT_POLICY = Path(
    "/home/ubt2204/work/planc_repro/robot_rl/logs/rsl_rl/g1/"
    "2026-07-09_17-27-32/exported/policy.pt"
)

FULL_DEFAULT = np.array(
    [
        -0.25,
        0.0,
        0.0,
        0.46,
        -0.25,
        0.0,
        -0.25,
        0.0,
        0.0,
        0.46,
        -0.25,
        0.0,
        0.0,
        0.0,
        0.0,
        0.07,
        0.24,
        0.0,
        1.39,
        0.0,
        0.0,
        0.0,
        0.07,
        -0.24,
        0.0,
        1.39,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)

KP = np.array(
    [
        100,
        100,
        100,
        150,
        40,
        40,
        100,
        100,
        100,
        150,
        40,
        40,
        100,
        200,
        200,
        100,
        100,
        50,
        50,
        40,
        20,
        20,
        100,
        100,
        50,
        50,
        40,
        20,
        20,
    ],
    dtype=np.float64,
)
KD = np.array(
    [
        2,
        2,
        2,
        4,
        2,
        2,
        2,
        2,
        2,
        4,
        2,
        2,
        2,
        4,
        4,
        2,
        2,
        2,
        2,
        1,
        0.5,
        0.5,
        2,
        2,
        2,
        2,
        1,
        0.5,
        0.5,
    ],
    dtype=np.float64,
)

MAPS = {
    # PhysX/Isaac breadth-first DOF order with locked joints removed.
    "isaac_bfs": [0, 6, 12, 1, 7, 2, 8, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25],
    # The literal 21 names in env.yaml / URDF / SDK order.
    "sdk_active": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 22, 23, 24, 25],
    # Older transfer order that interleaves arms before leg yaw/knee.
    "old_transfer": [0, 6, 12, 1, 7, 15, 22, 2, 8, 16, 23, 3, 9, 17, 24, 4, 10, 18, 25, 5, 11],
}
LOCKED = [13, 14, 19, 20, 21, 26, 27, 28]


@dataclass
class Result:
    map_name: str
    cmd_x: float
    height_mode: str
    height_value: float
    seconds: float
    fell: bool
    x: float
    y: float
    z: float
    upright: float
    raw_min: float
    raw_max: float
    raw_mean: float
    height_min: float
    height_max: float
    height_mean: float
    ctrl_abs_max: float


def update_command(current: float, target: float, dt: float, slew_rate: float | None) -> float:
    if slew_rate is None or slew_rate <= 0.0:
        return target
    delta = target - current
    max_delta = slew_rate * dt
    return current + float(np.clip(delta, -max_delta, max_delta))


def quat_conj_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    # q is wxyz. Returns q^-1 * v * q.
    w, x, y, z = q
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v - w * t + np.cross(qv, t)


def body_z_world(data: mujoco.MjData, body_id: int) -> float:
    return float(data.xmat[body_id].reshape(3, 3)[2, 2])


def cast_world_hit_z(model: mujoco.MjModel, data: mujoco.MjData, origin: np.ndarray) -> float:
    ray = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    pos = origin.astype(np.float64).copy()
    for _ in range(32):
        geom_id = np.array([-1], dtype=np.int32)
        distance = float(mujoco.mj_ray(model, data, pos, ray, None, 1, -1, geom_id))
        if distance <= 0.0 or geom_id[0] < 0:
            return 0.0
        hit_z = float(pos[2] - distance)
        body_id = int(model.geom_bodyid[int(geom_id[0])])
        if body_id == 0:
            return hit_z
        pos = pos + ray * (distance + 0.02)
    return 0.0


def compute_height_scan(
    args: argparse.Namespace,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    height_value: float,
) -> np.ndarray:
    if args.height_mode == "constant":
        return np.full(121, height_value, dtype=np.float32)

    xpos = np.array(data.xpos[body_id], dtype=np.float64)
    xmat = np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    yaw = math.atan2(float(xmat[1, 0]), float(xmat[0, 0]))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    xs = np.arange(-0.5, 0.5 + 1.0e-9, 0.1, dtype=np.float64)
    ys = np.arange(-0.5, 0.5 + 1.0e-9, 0.1, dtype=np.float64)
    values: list[float] = []
    for local_y in ys:
        for local_x in xs:
            origin = np.array(
                [
                    xpos[0] + cos_yaw * local_x - sin_yaw * local_y,
                    xpos[1] + sin_yaw * local_x + cos_yaw * local_y,
                    xpos[2] + 20.0,
                ],
                dtype=np.float64,
            )
            hit_z = cast_world_hit_z(model, data, origin)
            values.append(float(np.clip(xpos[2] - hit_z - 0.5, -2.0, 2.0)))
    return np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)


def run_once(args: argparse.Namespace, map_name: str, cmd_x: float, height_value: float) -> Result:
    active = np.array(MAPS[map_name], dtype=np.int32)
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)

    if args.timestep is not None:
        model.opt.timestep = args.timestep

    qadr = []
    dadr = []
    for motor_id in range(model.nu):
        joint_id = int(model.actuator_trnid[motor_id, 0])
        qadr.append(int(model.jnt_qposadr[joint_id]))
        dadr.append(int(model.jnt_dofadr[joint_id]))

    data.qpos[0:3] = [0.0, 0.0, args.base_z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for motor_id, qpos_adr in enumerate(qadr):
        data.qpos[qpos_adr] = FULL_DEFAULT[motor_id]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    policy = torch.jit.load(str(args.policy), map_location="cpu")
    policy.eval()

    imu_quat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
    imu_gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    imu_quat_adr = int(model.sensor_adr[imu_quat_id])
    imu_gyro_adr = int(model.sensor_adr[imu_gyro_id])
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    full_target = FULL_DEFAULT.copy()
    last_raw = np.zeros(len(active), dtype=np.float32)
    raw_min = 0.0
    raw_max = 0.0
    raw_sum = 0.0
    raw_count = 0
    height_min = float("inf")
    height_max = float("-inf")
    height_sum = 0.0
    height_count = 0
    ctrl_abs_max = 0.0
    cmd_current = cmd_x
    blend_start_target: np.ndarray | None = None
    policy_timer = 0.0
    policy_dt = args.policy_dt
    sim_steps = int(args.duration / model.opt.timestep)
    fell = False

    for _ in range(sim_steps):
        if policy_timer <= 1.0e-12:
            cmd_target = cmd_x
            if args.stop_after is not None and data.time >= args.stop_after:
                cmd_target = 0.0
            cmd_current = update_command(
                cmd_current,
                cmd_target,
                policy_dt,
                args.command_slew_rate,
            )
            q = np.array(data.sensordata[imu_quat_adr : imu_quat_adr + 4], dtype=np.float64)
            gyro = np.array(data.sensordata[imu_gyro_adr : imu_gyro_adr + 3], dtype=np.float32)
            gravity = quat_conj_rotate(q, np.array([0.0, 0.0, -1.0], dtype=np.float64)).astype(np.float32)
            joint_pos = np.array([data.qpos[qadr[i]] for i in active], dtype=np.float32)
            joint_vel = np.array([data.qvel[dadr[i]] for i in active], dtype=np.float32)
            default_active = FULL_DEFAULT[active].astype(np.float32)
            height = compute_height_scan(args, model, data, pelvis_id, height_value)
            obs = np.concatenate(
                [
                    gyro,
                    gravity,
                    np.array([cmd_current, 0.0, 0.0], dtype=np.float32) * 2.0,
                    joint_pos - default_active,
                    joint_vel * 0.05,
                    last_raw,
                    height,
                ]
            ).astype(np.float32)
            with torch.no_grad():
                raw = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).cpu().numpy().astype(np.float32)
            if args.raw_clip is not None:
                raw = np.clip(raw, -args.raw_clip, args.raw_clip)
            last_raw = raw
            full_target[:] = FULL_DEFAULT
            full_target[active] = FULL_DEFAULT[active] + args.action_scale * raw.astype(np.float64)
            full_target[LOCKED] = FULL_DEFAULT[LOCKED]
            if (
                args.hold_default_after_stop
                and args.stop_after is not None
                and data.time >= args.stop_after + args.hold_default_delay
            ):
                full_target[:] = FULL_DEFAULT
            if (
                args.blend_default_after_stop
                and args.stop_after is not None
                and data.time >= args.stop_after
            ):
                if blend_start_target is None:
                    blend_start_target = full_target.copy()
                alpha = min(1.0, (data.time - args.stop_after) / max(args.blend_duration, 1.0e-6))
                full_target[:] = (1.0 - alpha) * blend_start_target + alpha * FULL_DEFAULT
            raw_min = min(raw_min, float(raw.min()))
            raw_max = max(raw_max, float(raw.max()))
            raw_sum += float(raw.mean())
            raw_count += 1
            height_min = min(height_min, float(height.min()))
            height_max = max(height_max, float(height.max()))
            height_sum += float(height.mean())
            height_count += 1
            policy_timer += policy_dt

        for motor_id in range(model.nu):
            q = data.qpos[qadr[motor_id]]
            dq = data.qvel[dadr[motor_id]]
            ctrl = KP[motor_id] * (full_target[motor_id] - q) - KD[motor_id] * dq
            data.ctrl[motor_id] = ctrl
        ctrl_abs_max = max(ctrl_abs_max, float(np.max(np.abs(data.ctrl))))

        mujoco.mj_step(model, data)
        policy_timer -= model.opt.timestep
        upright = body_z_world(data, pelvis_id)
        if data.qpos[2] < args.fall_z or upright < args.fall_upright:
            fell = True
            break

    seconds = float(data.time)
    upright = body_z_world(data, pelvis_id)
    return Result(
        map_name=map_name,
        cmd_x=cmd_x,
        height_mode=args.height_mode,
        height_value=height_value,
        seconds=seconds,
        fell=fell,
        x=float(data.qpos[0]),
        y=float(data.qpos[1]),
        z=float(data.qpos[2]),
        upright=upright,
        raw_min=raw_min,
        raw_max=raw_max,
        raw_mean=raw_sum / max(1, raw_count),
        height_min=height_min if height_count else 0.0,
        height_max=height_max if height_count else 0.0,
        height_mean=height_sum / max(1, height_count),
        ctrl_abs_max=ctrl_abs_max,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--map", choices=list(MAPS) + ["all"], default="all")
    parser.add_argument("--cmd-x", type=float, default=0.6)
    parser.add_argument("--stop-after", type=float, default=None)
    parser.add_argument("--command-slew-rate", type=float, default=None)
    parser.add_argument("--hold-default-after-stop", action="store_true")
    parser.add_argument("--hold-default-delay", type=float, default=0.0)
    parser.add_argument("--blend-default-after-stop", action="store_true")
    parser.add_argument("--blend-duration", type=float, default=3.0)
    parser.add_argument("--height-mode", choices=["constant", "true"], default="constant")
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--policy-dt", type=float, default=0.02)
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--base-z", type=float, default=0.785)
    parser.add_argument("--fall-z", type=float, default=0.35)
    parser.add_argument("--fall-upright", type=float, default=0.55)
    parser.add_argument("--raw-clip", type=float, default=None)
    parser.add_argument("--timestep", type=float, default=None)
    args = parser.parse_args()

    maps = list(MAPS) if args.map == "all" else [args.map]
    print(
        "map,cmd_x,height_mode,height,seconds,fell,x,y,z,upright,"
        "raw_min,raw_max,raw_mean,height_min,height_max,height_mean,ctrl_abs_max"
    )
    for map_name in maps:
        res = run_once(args, map_name, args.cmd_x, args.height)
        print(
            f"{res.map_name},{res.cmd_x:.3f},{res.height_mode},{res.height_value:.3f},"
            f"{res.seconds:.3f},{int(res.fell)},{res.x:.3f},{res.y:.3f},"
            f"{res.z:.3f},{res.upright:.3f},{res.raw_min:.3f},"
            f"{res.raw_max:.3f},{res.raw_mean:.3f},{res.height_min:.3f},"
            f"{res.height_max:.3f},{res.height_mean:.3f},{res.ctrl_abs_max:.1f}"
        )


if __name__ == "__main__":
    main()
