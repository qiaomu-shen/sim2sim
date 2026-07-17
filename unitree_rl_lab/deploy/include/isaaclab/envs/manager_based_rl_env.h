// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include "isaaclab/utils/utils.h"

namespace isaaclab
{

class ObservationManager;
class ActionManager;

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = cfg["step_dt"].as<float>();
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<int>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        robot->update();

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        initializing_observation_manager = true;
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
        initializing_observation_manager = false;
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        has_last_nonzero_base_velocity_command = false;
        last_nonzero_base_velocity_command = {0.0f, 0.0f, 0.0f};
        robot->update();
        action_manager->reset();
        observation_manager->reset();
    }

    void step()
    {
        const bool use_training_step_semantics =
            cfg["use_training_step_semantics"].as<bool>(false);
        if (!use_training_step_semantics) {
            episode_length += 1;
        }
        robot->update();
        auto obs = observation_manager->compute();
        auto action = hold_default_on_zero_command()
            ? std::vector<float>(
                static_cast<size_t>(action_manager->total_action_dim()),
                0.0f)
            : alg->act(obs);
        action_manager->process_action(action);
        if (use_training_step_semantics) {
            episode_length += 1;
        }
    }

    bool hold_default_on_zero_command() const
    {
        const auto command_cfg = cfg["commands"]["base_velocity"];
        if (!command_cfg ||
            !command_cfg["hold_default_on_zero_command"].as<bool>(false)) {
            return false;
        }

        auto* joystick = robot->data.joystick;
        if (joystick == nullptr) {
            return false;
        }

        const float deadzone = command_cfg["deadzone"].as<float>(0.0f);
        const bool zero_on_deadzone =
            command_cfg["zero_command_on_deadzone"].as<bool>(false);
        const bool snap_to_limit =
            command_cfg["snap_to_limit_on_input"].as<bool>(false);

        const float raw_x = joystick->ly();
        const float raw_y = -joystick->lx();
        const float raw_yaw = -joystick->rx();
        if (zero_on_deadzone &&
            std::abs(raw_x) < deadzone &&
            std::abs(raw_y) < deadzone &&
            std::abs(raw_yaw) < deadzone) {
            return true;
        }

        std::array<float, 3> command = {0.0f, 0.0f, 0.0f};
        const std::array<float, 3> raw = {raw_x, raw_y, raw_yaw};
        const std::array<const char*, 3> keys = {"lin_vel_x", "lin_vel_y", "ang_vel_z"};
        const auto ranges = command_cfg["ranges"];
        for (size_t i = 0; i < command.size(); ++i) {
            const float lower = ranges[keys[i]][0].as<float>();
            const float upper = ranges[keys[i]][1].as<float>();
            if (snap_to_limit) {
                if (raw[i] > deadzone && upper > 0.0f) {
                    command[i] = upper;
                } else if (raw[i] < -deadzone && lower < 0.0f) {
                    command[i] = lower;
                }
            } else {
                command[i] = std::clamp(raw[i], lower, upper);
            }
            if (std::abs(command[i]) < deadzone) {
                command[i] = 0.0f;
            }
        }

        return std::abs(command[0]) < deadzone &&
            std::abs(command[1]) < deadzone &&
            std::abs(command[2]) < deadzone;
    }

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    long episode_length = 0;
    float global_phase = 0.0f;
    bool initializing_observation_manager = false;
    std::array<float, 3> last_nonzero_base_velocity_command = {0.0f, 0.0f, 0.0f};
    bool has_last_nonzero_base_velocity_command = false;
};

};
