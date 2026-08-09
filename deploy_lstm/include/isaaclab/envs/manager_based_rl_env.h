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
#include <functional>
#include <iostream>
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
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        robot->update();

        // Reset stair_latent delta-feature caches so the first
        // observation frame after a state change produces zero deltas
        // instead of garbage from a previous policy invocation.
        robot->data.stair_latent_cache_valid = false;
        robot->data.prev_toe_vel_valid = false;
        if (robot->data.foot_event_summary.size() != 80) {
            robot->data.foot_event_summary.assign(80, 0.0f);
        } else {
            std::fill(
                robot->data.foot_event_summary.begin(),
                robot->data.foot_event_summary.end(),
                0.0f);
        }

        action_manager->reset();
        observation_manager->reset();
        if (alg) {
            alg->reset();
        }
        if (reset_callback) {
            reset_callback();
        }
    }

    void step(const std::function<void()>& before_observation = nullptr)
    {
        const bool use_training_step_semantics =
            cfg["use_training_step_semantics"].as<bool>(false);
        if (!use_training_step_semantics) {
            // Preserve the legacy deployment timing used by existing policies.
            episode_length += 1;
        }
        robot->update();
        if (before_observation) {
            before_observation();
        }
        auto obs = observation_manager->compute();
        auto action = alg->act(obs);
        action_manager->process_action(action);
        if (use_training_step_semantics) {
            // mjlab consumes the reset observation at policy step 0 and
            // advances the episode counter afterwards.
            episode_length += 1;
        }
    }

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    std::function<void()> reset_callback;
    long episode_length = 0;
    float global_phase = 0.0f;
};

};
