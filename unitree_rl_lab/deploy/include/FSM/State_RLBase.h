// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <vector>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        entry_posture_active_ = false;
        policy_thread_running = false;

        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }
        apply_locked_joint_commands(true);

        env->robot->update();

        if (configure_entry_posture()) {
            return;
        }
        start_policy_thread();
    }

    void start_policy_thread()
    {
        if (policy_thread_running) {
            return;
        }

        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        entry_posture_active_ = false;
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    static float vector_value_or(
        const std::vector<float>& values,
        const size_t index,
        const float fallback)
    {
        return index < values.size() ? values[index] : fallback;
    }

    static std::vector<float> float_vector_from_cfg(
        const YAML::Node& cfg,
        const char* key,
        const size_t size,
        const float fallback)
    {
        std::vector<float> values(size, fallback);
        if (!cfg || !cfg[key]) {
            return values;
        }

        try {
            const auto parsed = cfg[key].as<std::vector<float>>();
            for (size_t i = 0; i < values.size() && i < parsed.size(); ++i) {
                values[i] = parsed[i];
            }
        } catch (const std::exception&) {
        }
        return values;
    }

    void apply_locked_joint_commands(const bool set_gains)
    {
        const auto cfg = env->cfg["locked_joints"];
        if (!cfg || !cfg["ids"]) {
            return;
        }

        std::vector<int> ids;
        try {
            ids = cfg["ids"].as<std::vector<int>>();
        } catch (const std::exception&) {
            return;
        }

        const auto positions = float_vector_from_cfg(cfg, "positions", ids.size(), 0.0f);
        const auto kp = float_vector_from_cfg(cfg, "stiffness", ids.size(), 0.0f);
        const auto kd = float_vector_from_cfg(cfg, "damping", ids.size(), 0.0f);

        for (size_t i = 0; i < ids.size(); ++i) {
            const int motor_id = ids[i];
            auto& motor_cmd = lowcmd->msg_.motor_cmd()[motor_id];
            if (set_gains) {
                motor_cmd.kp() = vector_value_or(kp, i, 0.0f);
                motor_cmd.kd() = vector_value_or(kd, i, 0.0f);
            }
            motor_cmd.q() = vector_value_or(positions, i, 0.0f);
            motor_cmd.dq() = 0;
            motor_cmd.tau() = 0;
        }
    }

    bool configure_entry_posture()
    {
        entry_posture_q0_.clear();
        entry_posture_target_.clear();

        const auto cfg = env->cfg["entry_posture"];
        if (!cfg || !cfg["enabled"].as<bool>(false)) {
            return false;
        }

        entry_posture_duration_s_ =
            std::max(0.0f, cfg["duration_s"].as<float>(0.0f));
        if (entry_posture_duration_s_ <= 0.0f) {
            return false;
        }

        const auto& joint_ids = env->robot->data.joint_ids_map;
        entry_posture_q0_.assign(joint_ids.size(), 0.0f);
        {
            std::lock_guard<std::mutex> lock(lowstate->mutex_);
            for (size_t i = 0; i < joint_ids.size(); ++i) {
                const int motor_id = static_cast<int>(joint_ids[i]);
                entry_posture_q0_[i] =
                    lowstate->msg_.motor_state()[motor_id].q();
            }
        }

        entry_posture_target_.assign(
            env->robot->data.default_joint_pos.data(),
            env->robot->data.default_joint_pos.data() +
                env->robot->data.default_joint_pos.size());
        if (cfg["positions"]) {
            try {
                const auto positions = cfg["positions"].as<std::vector<float>>();
                if (positions.size() == joint_ids.size()) {
                    entry_posture_target_ = positions;
                }
            } catch (const std::exception&) {
            }
        }

        entry_posture_t0_ = std::chrono::steady_clock::now();
        entry_posture_active_ = true;
        return true;
    }

    bool run_entry_posture()
    {
        if (!entry_posture_active_) {
            return false;
        }

        const auto now = std::chrono::steady_clock::now();
        const float elapsed_s = std::chrono::duration<float>(
            now - entry_posture_t0_).count();
        const float alpha = std::clamp(
            elapsed_s / std::max(entry_posture_duration_s_, 1e-6f),
            0.0f,
            1.0f);

        const auto& joint_ids = env->robot->data.joint_ids_map;
        const size_t count = std::min(
            joint_ids.size(),
            std::min(entry_posture_q0_.size(), entry_posture_target_.size()));
        for (size_t i = 0; i < count; ++i) {
            const int motor_id = static_cast<int>(joint_ids[i]);
            auto& motor_cmd = lowcmd->msg_.motor_cmd()[motor_id];
            motor_cmd.q() = (1.0f - alpha) * entry_posture_q0_[i] +
                alpha * entry_posture_target_[i];
            motor_cmd.dq() = 0;
            motor_cmd.tau() = 0;
        }
        apply_locked_joint_commands(false);

        if (alpha >= 1.0f) {
            entry_posture_active_ = false;
            start_policy_thread();
        }
        return true;
    }

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
    bool entry_posture_active_ = false;
    float entry_posture_duration_s_ = 0.0f;
    std::chrono::steady_clock::time_point entry_posture_t0_;
    std::vector<float> entry_posture_q0_;
    std::vector<float> entry_posture_target_;
};

REGISTER_FSM(State_RLBase)
