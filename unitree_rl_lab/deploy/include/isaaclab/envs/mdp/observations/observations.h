// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(height_scan)
{
    int size = 121;
    float fill_value = 0.0f;
    float max_age_s = -1.0f;
    bool require_bridge = false;
    bool allow_missing_during_init = false;
    std::string bridge_file;

    try {
        if (params["size"].IsDefined()) {
            size = params["size"].as<int>();
        }
        if (params["fill_value"].IsDefined()) {
            fill_value = params["fill_value"].as<float>();
        }
        if (params["max_age_s"].IsDefined()) {
            max_age_s = params["max_age_s"].as<float>();
        }
        if (params["require_bridge"].IsDefined()) {
            require_bridge = params["require_bridge"].as<bool>();
        }
        if (params["allow_missing_during_init"].IsDefined()) {
            allow_missing_during_init = params["allow_missing_during_init"].as<bool>();
        }
        if (params["bridge_file"].IsDefined()) {
            bridge_file = params["bridge_file"].as<std::string>();
        }
        if (params["values"].IsDefined()) {
            const auto values = params["values"].as<std::vector<float>>();
            if (!values.empty()) {
                return values;
            }
        }
    } catch (const std::exception&) {
    }

    const auto fallback = [&](const std::string& reason) -> std::vector<float> {
        const bool init_fallback_allowed =
            allow_missing_during_init && env != nullptr && env->initializing_observation_manager;
        if (require_bridge && !init_fallback_allowed) {
            throw std::runtime_error("height_scan bridge required but unavailable: " + reason);
        }
        return std::vector<float>(std::max(0, size), fill_value);
    };

    if (!bridge_file.empty()) {
        if (max_age_s >= 0.0f) {
            try {
                const auto write_time = std::filesystem::last_write_time(bridge_file);
                const auto now = std::filesystem::file_time_type::clock::now();
                const float age_s =
                    std::chrono::duration<float>(now - write_time).count();
                if (age_s > max_age_s) {
                    return fallback("stale file " + bridge_file);
                }
            } catch (const std::exception&) {
                return fallback("cannot stat " + bridge_file);
            }
        }

        std::ifstream input(bridge_file, std::ios::binary);
        if (input) {
            std::string header;
            if (std::getline(input, header)) {
                std::istringstream header_stream(header);
                std::string magic;
                int count = 0;
                header_stream >> magic >> count;
                if (magic == "UTHEIGHT1" && count == size && count > 0) {
                    std::vector<float> values(static_cast<size_t>(count), fill_value);
                    input.read(
                        reinterpret_cast<char*>(values.data()),
                        static_cast<std::streamsize>(values.size() * sizeof(float)));
                    if (input.gcount() == static_cast<std::streamsize>(values.size() * sizeof(float))) {
                        return values;
                    }
                }
            }
            return fallback("invalid file " + bridge_file);
        }
        return fallback("cannot open " + bridge_file);
    }

    return fallback("bridge_file is empty");
}

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3);
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];
    const float deadzone =
        env->cfg["commands"]["base_velocity"]["deadzone"].as<float>(0.0f);
    const bool zero_on_deadzone =
        env->cfg["commands"]["base_velocity"]["zero_command_on_deadzone"].as<bool>(false);
    const bool snap_to_limit =
        env->cfg["commands"]["base_velocity"]["snap_to_limit_on_input"].as<bool>(false);
    const bool keep_last_nonzero_on_zero =
        env->cfg["commands"]["base_velocity"]["keep_last_nonzero_on_zero_command"].as<bool>(false);

    const float raw_x = joystick->ly();
    const float raw_y = -joystick->lx();
    const float raw_yaw = -joystick->rx();

    if (zero_on_deadzone &&
        std::abs(raw_x) < deadzone &&
        std::abs(raw_y) < deadzone &&
        std::abs(raw_yaw) < deadzone) {
        if (keep_last_nonzero_on_zero && env->has_last_nonzero_base_velocity_command) {
            return std::vector<float>(
                env->last_nonzero_base_velocity_command.begin(),
                env->last_nonzero_base_velocity_command.end());
        }
        return obs;
    }

    const std::array<float, 3> raw = {raw_x, raw_y, raw_yaw};
    const std::array<const char*, 3> keys = {"lin_vel_x", "lin_vel_y", "ang_vel_z"};
    for (size_t i = 0; i < obs.size(); ++i) {
        const float lower = cfg[keys[i]][0].as<float>();
        const float upper = cfg[keys[i]][1].as<float>();
        if (snap_to_limit) {
            if (raw[i] > deadzone && upper > 0.0f) {
                obs[i] = upper;
            } else if (raw[i] < -deadzone && lower < 0.0f) {
                obs[i] = lower;
            }
        } else {
            obs[i] = std::clamp(raw[i], lower, upper);
        }
    }

    for (float & command : obs) {
        if (std::abs(command) < deadzone) {
            command = 0.0f;
        }
    }

    if (std::abs(obs[0]) >= deadzone ||
        std::abs(obs[1]) >= deadzone ||
        std::abs(obs[2]) >= deadzone) {
        env->last_nonzero_base_velocity_command = {obs[0], obs[1], obs[2]};
        env->has_last_nonzero_base_velocity_command = true;
    }

    return obs;
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);
    return obs;
}

}
}
