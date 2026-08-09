// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "foot_event_memory.h"
#include "foot_event_observer.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "touchdown_detector.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        set_active_joint_gains();
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

            env->reset();
            wait_for_camera_startup();

            const int diagnostics_interval_steps =
                env->cfg["diagnostics"]["log_interval_steps"].as<int>(0);
            prepare_diagnostics_run_dir();
            open_touchdown_log();
            open_touchdown_event_log();
            open_foot_event_observer();

            // Initialize timing after any camera startup wait.
            auto sleepTill = clock::now() + dt;

            while (policy_thread_running)
            {
                env->step([this] { update_foot_event_memory_before_policy(); });
                write_touchdown_log_row();

                if (diagnostics_interval_steps > 0 &&
                    env->episode_length % diagnostics_interval_steps == 0)
                {
                    const auto command =
                        isaaclab::mdp::velocity_commands(env.get(), YAML::Node{});
                    const auto & joystick = env->robot->data.joystick;
                    const auto & action = env->action_manager->action();
                    const auto action_extrema =
                        std::minmax_element(action.begin(), action.end());
                    const float action_min = action.empty()
                        ? 0.0f : *action_extrema.first;
                    const float action_max = action.empty()
                        ? 0.0f : *action_extrema.second;

                    bool has_height_scan = false;
                    float height_scan_min = 0.0f;
                    float height_scan_max = 0.0f;
                    try {
                        const auto height_cfg = env->cfg["observations"]["height_scan"];
                        if (height_cfg && height_cfg["params"]) {
                            const auto height_scan =
                                isaaclab::mdp::height_scan(env.get(), height_cfg["params"]);
                            const auto height_extrema =
                                std::minmax_element(height_scan.begin(), height_scan.end());
                            if (!height_scan.empty()) {
                                has_height_scan = true;
                                height_scan_min = *height_extrema.first;
                                height_scan_max = *height_extrema.second;
                            }
                        }
                    } catch (const std::exception&) {
                    }

                    if (has_height_scan) {
                        spdlog::info(
                            "[{}] step={} raw_stick(ly,lx,rx)=[{:.4f}, {:.4f}, {:.4f}] "
                            "policy_command(vx,vy,wz)=[{:.4f}, {:.4f}, {:.4f}] "
                            "height_scan[min,max]=[{:.4f}, {:.4f}] "
                            "policy_action[min,max]=[{:.4f}, {:.4f}]",
                            getStateString(),
                            env->episode_length,
                            joystick->ly(), joystick->lx(), joystick->rx(),
                            command[0], command[1], command[2],
                            height_scan_min, height_scan_max,
                            action_min, action_max
                        );
                    } else {
                    spdlog::info(
                        "[{}] step={} raw_stick(ly,lx,rx)=[{:.4f}, {:.4f}, {:.4f}] "
                        "policy_command(vx,vy,wz)=[{:.4f}, {:.4f}, {:.4f}] "
                        "policy_action[min,max]=[{:.4f}, {:.4f}]",
                        getStateString(),
                        env->episode_length,
                        joystick->ly(), joystick->lx(), joystick->rx(),
                        command[0], command[1], command[2],
                        action_min, action_max
                    );
                    }
                }

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
            close_foot_event_observer();
            close_touchdown_event_log();
            close_touchdown_log();
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
    static constexpr std::array<const char*, 29> kJointColumnNames = {{
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
        "left_knee", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
        "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch",
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
        "left_elbow", "left_wrist_roll", "left_wrist_pitch",
        "left_wrist_yaw",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
        "right_elbow", "right_wrist_roll", "right_wrist_pitch",
        "right_wrist_yaw",
    }};

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

    static float cfg_float(
        const YAML::Node& cfg,
        const char* key,
        const float fallback)
    {
        if (!cfg || !cfg[key]) {
            return fallback;
        }
        return cfg[key].as<float>(fallback);
    }

    static int cfg_int(
        const YAML::Node& cfg,
        const char* key,
        const int fallback)
    {
        if (!cfg || !cfg[key]) {
            return fallback;
        }
        return cfg[key].as<int>(fallback);
    }

    static bool cfg_bool(
        const YAML::Node& cfg,
        const char* key,
        const bool fallback)
    {
        if (!cfg || !cfg[key]) {
            return fallback;
        }
        return cfg[key].as<bool>(fallback);
    }

    void set_active_joint_gains()
    {
        const auto& joint_ids = env->robot->data.joint_ids_map;
        const auto& stiffness = env->robot->data.joint_stiffness;
        const auto& damping = env->robot->data.joint_damping;
        const size_t count = std::min(joint_ids.size(), stiffness.size());

        for (size_t i = 0; i < count; ++i) {
            const int motor_id = joint_ids[i];
            lowcmd->msg_.motor_cmd()[motor_id].kp() = stiffness[i];
            lowcmd->msg_.motor_cmd()[motor_id].kd() =
                i < damping.size() ? damping[i] : 0.0f;
            lowcmd->msg_.motor_cmd()[motor_id].dq() = 0;
            lowcmd->msg_.motor_cmd()[motor_id].tau() = 0;
        }
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
        entry_posture_active_ = false;
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
                entry_posture_q0_[i] =
                    lowstate->msg_.motor_state()[joint_ids[i]].q();
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
                } else {
                    spdlog::warn(
                        "[{}] entry_posture.positions size {} does not match active joints {}; using default_joint_pos",
                        getStateString(), positions.size(), joint_ids.size());
                }
            } catch (const std::exception& e) {
                spdlog::warn(
                    "[{}] failed to parse entry_posture.positions: {}; using default_joint_pos",
                    getStateString(), e.what());
            }
        }

        entry_posture_start_ = std::chrono::steady_clock::now();
        entry_posture_active_ = true;
        spdlog::info(
            "[{}] Entry posture blend enabled: duration={:.2f}s active_joints={}",
            getStateString(), entry_posture_duration_s_, joint_ids.size());
        return true;
    }

    bool run_entry_posture()
    {
        if (!entry_posture_active_) {
            return false;
        }

        const auto now = std::chrono::steady_clock::now();
        const float elapsed =
            std::chrono::duration<float>(now - entry_posture_start_).count();
        const float alpha = entry_posture_duration_s_ <= 0.0f
            ? 1.0f
            : std::min(1.0f, elapsed / entry_posture_duration_s_);

        const auto& joint_ids = env->robot->data.joint_ids_map;
        const size_t count = std::min(
            joint_ids.size(),
            std::min(entry_posture_q0_.size(), entry_posture_target_.size()));
        for (size_t i = 0; i < count; ++i) {
            auto& motor_cmd = lowcmd->msg_.motor_cmd()[joint_ids[i]];
            motor_cmd.q() = entry_posture_q0_[i] +
                alpha * (entry_posture_target_[i] - entry_posture_q0_[i]);
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

    static void append_vec3_header(std::ofstream& stream, const std::string& prefix)
    {
        stream << ',' << prefix << "_x"
               << ',' << prefix << "_y"
               << ',' << prefix << "_z";
    }

    static void append_vec3(std::ofstream& stream, const Eigen::Vector3f& value)
    {
        stream << ',' << value.x()
               << ',' << value.y()
               << ',' << value.z();
    }

    std::vector<float> action_scale() const
    {
        const int dim = env->action_manager->total_action_dim();
        std::vector<float> scale(dim, 1.0f);
        const auto actions_node = env->cfg["actions"];
        if (!actions_node) {
            return scale;
        }
        auto term_it = actions_node.begin();
        if (term_it == actions_node.end()) {
            return scale;
        }
        const auto action_cfg = term_it->second;
        if (!action_cfg["scale"].IsNull()) {
            const auto cfg_scale = action_cfg["scale"].as<std::vector<float>>();
            for (int i = 0; i < dim && i < static_cast<int>(cfg_scale.size()); ++i) {
                scale[i] = cfg_scale[i];
            }
        }
        return scale;
    }

    std::vector<float> action_offset() const
    {
        const int dim = env->action_manager->total_action_dim();
        std::vector<float> offset(dim, 0.0f);
        const auto actions_node = env->cfg["actions"];
        if (!actions_node) {
            return offset;
        }
        auto term_it = actions_node.begin();
        if (term_it == actions_node.end()) {
            return offset;
        }
        const auto action_cfg = term_it->second;
        if (!action_cfg["offset"].IsNull()) {
            const auto cfg_offset = action_cfg["offset"].as<std::vector<float>>();
            for (int i = 0; i < dim && i < static_cast<int>(cfg_offset.size()); ++i) {
                offset[i] = cfg_offset[i];
            }
        }
        return offset;
    }

    float gait_period() const
    {
        const auto observations = env->cfg["observations"];
        const auto actor_obs = observations["actor_obs"];
        if (actor_obs && actor_obs["gait_phase"] &&
            actor_obs["gait_phase"]["params"] &&
            actor_obs["gait_phase"]["params"]["period"]) {
            return actor_obs["gait_phase"]["params"]["period"].as<float>(0.6f);
        }
        if (observations && observations["gait_phase"] &&
            observations["gait_phase"]["params"] &&
            observations["gait_phase"]["params"]["period"]) {
            return observations["gait_phase"]["params"]["period"].as<float>(0.6f);
        }
        return 0.6f;
    }

    float current_gait_phase() const
    {
        const float period = std::max(gait_period(), 1.0e-6f);
        if (env->cfg["use_training_step_semantics"].as<bool>(false)) {
            return std::fmod(static_cast<float>(env->episode_length) * env->step_dt,
                             period) / period;
        }
        return env->global_phase;
    }

    static std::string make_run_timestamp()
    {
        const auto now = std::chrono::system_clock::now();
        const std::time_t time = std::chrono::system_clock::to_time_t(now);
        std::tm local_time{};
        localtime_r(&time, &local_time);

        char buffer[32] = {};
        std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &local_time);
        return buffer;
    }

    static std::filesystem::path strip_leading_log_component(
        const std::filesystem::path& path)
    {
        std::filesystem::path stripped;
        bool skipped = false;
        for (const auto& part : path) {
            if (!skipped && part == std::filesystem::path("log")) {
                skipped = true;
                continue;
            }
            stripped /= part;
        }
        return stripped.empty() ? path.filename() : stripped;
    }

    static std::filesystem::path unique_timestamped_dir(
        const std::filesystem::path& root)
    {
        const std::string stamp = make_run_timestamp();
        std::filesystem::path candidate = root / stamp;
        int suffix = 2;
        while (std::filesystem::exists(candidate)) {
            candidate = root / (stamp + "_" + std::to_string(suffix));
            suffix += 1;
        }
        return candidate;
    }

    void prepare_diagnostics_run_dir()
    {
        diagnostics_run_dir_.clear();

        const auto diagnostics_cfg = env->cfg["diagnostics"];
        const auto run_cfg = diagnostics_cfg["run_directory"];
        const bool enabled = !run_cfg || !run_cfg["enabled"] ||
                             run_cfg["enabled"].as<bool>(true);
        if (!enabled) {
            return;
        }

        std::filesystem::path root =
            run_cfg && run_cfg["root"]
                ? run_cfg["root"].as<std::string>()
                : "log";
        if (root.is_relative()) {
            root = param::proj_dir / root;
        }

        diagnostics_run_dir_ = unique_timestamped_dir(root);
        std::filesystem::create_directories(diagnostics_run_dir_);

        std::filesystem::path latest_file =
            run_cfg && run_cfg["latest_file"]
                ? run_cfg["latest_file"].as<std::string>()
                : "log/latest_run_dir.txt";
        if (latest_file.is_relative()) {
            latest_file = param::proj_dir / latest_file;
        }
        if (latest_file.has_parent_path()) {
            std::filesystem::create_directories(latest_file.parent_path());
        }

        std::ofstream latest(latest_file, std::ios::out | std::ios::trunc);
        if (latest) {
            latest << diagnostics_run_dir_.string() << '\n';
        } else {
            spdlog::warn("Failed to write diagnostics latest run pointer: {}",
                         latest_file.string());
        }

        spdlog::info("Diagnostics run directory: {}",
                     diagnostics_run_dir_.string());
    }

    std::filesystem::path diagnostics_path_from_cfg(
        const YAML::Node& cfg,
        const std::string& default_path) const
    {
        std::filesystem::path path = default_path;
        if (cfg && cfg["path"]) {
            path = cfg["path"].as<std::string>(default_path);
        }
        if (path.is_relative()) {
            if (!diagnostics_run_dir_.empty()) {
                return diagnostics_run_dir_ / strip_leading_log_component(path);
            }
            path = param::proj_dir / path;
        }
        return path;
    }

    std::filesystem::path model_path_from_cfg(
        const YAML::Node& cfg,
        const std::string& default_path) const
    {
        std::filesystem::path path = default_path;
        if (cfg && cfg["model_path"]) {
            path = cfg["model_path"].as<std::string>(default_path);
        }
        if (path.is_absolute()) {
            return path;
        }

        const std::array<std::filesystem::path, 3> bases = {{
            param::proj_dir,
            param::proj_dir.parent_path().parent_path().parent_path(),
            std::filesystem::current_path(),
        }};
        for (const auto& base : bases) {
            const auto candidate = base / path;
            if (std::filesystem::exists(candidate)) {
                return candidate;
            }
        }
        return bases[1] / path;
    }

    std::filesystem::path touchdown_log_path_from_cfg() const
    {
        const auto log_cfg = env->cfg["diagnostics"]["touchdown_log"];
        return diagnostics_path_from_cfg(log_cfg, "mjlab_g1_touchdown_log.csv");
    }

    void open_touchdown_log()
    {
        const auto log_cfg = env->cfg["diagnostics"]["touchdown_log"];
        touchdown_log_enabled_ = log_cfg["enabled"].as<bool>(false);
        touchdown_log_flush_interval_steps_ =
            std::max(1, log_cfg["flush_interval_steps"].as<int>(50));
        touchdown_log_stance_threshold_ =
            log_cfg["stance_threshold"].as<float>(0.56f);
        if (!touchdown_log_enabled_) {
            return;
        }

        const auto path = touchdown_log_path_from_cfg();
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        touchdown_log_.open(path, std::ios::out | std::ios::trunc);
        if (!touchdown_log_) {
            spdlog::error("Failed to open touchdown diagnostics log: {}", path.string());
            touchdown_log_enabled_ = false;
            return;
        }

        touchdown_log_.setf(std::ios::fixed);
        touchdown_log_ << std::setprecision(7);
        touchdown_log_path_ = path.string();
        touchdown_log_prev_valid_ = false;
        touchdown_log_prev_imu_valid_ = false;
        write_touchdown_log_header();
        spdlog::info("Touchdown diagnostics log enabled: {}", touchdown_log_path_);
    }

    void open_touchdown_event_log()
    {
        const auto event_cfg = env->cfg["diagnostics"]["touchdown_events"];
        touchdown_log_prev_valid_ = false;
        touchdown_log_prev_imu_valid_ = false;

        mjlab::diagnostics::TouchdownDetectorConfig cfg;
        cfg.enabled = event_cfg["enabled"].as<bool>(false);
        cfg.path = diagnostics_path_from_cfg(event_cfg, "log/touchdown_events.csv");
        cfg.flush_interval_events =
            std::max(1, event_cfg["flush_interval_events"].as<int>(10));
        cfg.score_threshold = event_cfg["score_threshold"].as<int>(3);
        cfg.require_motion_cue = event_cfg["require_motion_cue"].as<bool>(true);
        cfg.off_confirm_frames =
            std::max(1, event_cfg["off_confirm_frames"].as<int>(3));
        cfg.on_confirm_frames =
            std::max(1, event_cfg["on_confirm_frames"].as<int>(2));
        cfg.buffer_frames = std::max(8, event_cfg["buffer_frames"].as<int>(16));
        cfg.event_search_frames =
            std::max(1, event_cfg["event_search_frames"].as<int>(8));
        cfg.pre_window_frames =
            std::max(1, event_cfg["pre_window_frames"].as<int>(3));
        cfg.post_window_frames =
            std::max(1, event_cfg["post_window_frames"].as<int>(3));
        cfg.post_confirm_frames =
            std::max(0, event_cfg["post_confirm_frames"].as<int>(3));
        cfg.min_air_frames = std::max(
            1,
            static_cast<int>(std::ceil(
                event_cfg["min_air_time_s"].as<float>(0.08f) / env->step_dt
            ))
        );
        cfg.cooldown_frames = std::max(
            1,
            static_cast<int>(std::ceil(
                event_cfg["cooldown_s"].as<float>(0.20f) / env->step_dt
            ))
        );
        cfg.startup_ignore_s = event_cfg["startup_ignore_s"].as<double>(0.5);
        cfg.contact_z_threshold =
            event_cfg["contact_z_threshold"].as<float>(-0.74f);
        cfg.air_z_threshold = event_cfg["air_z_threshold"].as<float>(-0.70f);
        cfg.liftoff_vz_threshold =
            event_cfg["liftoff_vz_threshold"].as<float>(0.05f);
        cfg.vz_brake_down_threshold =
            event_cfg["vz_brake_down_threshold"].as<float>(-0.02f);
        cfg.vz_brake_abs_threshold =
            event_cfg["vz_brake_abs_threshold"].as<float>(0.08f);
        cfg.hspeed_threshold = event_cfg["hspeed_threshold"].as<float>(0.18f);
        cfg.tracking_delta_threshold =
            event_cfg["tracking_delta_threshold"].as<float>(0.03f);
        cfg.tau_delta_threshold =
            event_cfg["tau_delta_threshold"].as<float>(0.5f);
        cfg.imu_jerk_threshold =
            event_cfg["imu_jerk_threshold"].as<float>(80.0f);

        touchdown_detector_.configure(cfg);
        if (!cfg.enabled) {
            return;
        }
        if (touchdown_detector_.open()) {
            spdlog::info("Touchdown event log enabled: {}", cfg.path.string());
        } else {
            spdlog::error("Failed to open touchdown event log: {}", cfg.path.string());
        }
    }

    void open_foot_event_observer()
    {
        const auto diagnostics_cfg = env->cfg["diagnostics"];
        const auto observer_cfg =
            diagnostics_cfg ? diagnostics_cfg["foot_event_observer"] : YAML::Node{};
        const auto memory_cfg =
            diagnostics_cfg ? diagnostics_cfg["foot_event_memory"] : YAML::Node{};

        mjlab::diagnostics::FootEventObserverConfig cfg;
        configure_foot_event_memory(memory_cfg);
        if (!observer_cfg) {
            foot_event_observer_.configure(cfg);
            return;
        }

        const auto frame_cfg =
            observer_cfg["frame_log"] ? observer_cfg["frame_log"] : YAML::Node{};

        cfg.event_path = diagnostics_path_from_cfg(
            observer_cfg,
            "log/foot_event_observer_events.csv");
        cfg.frame_path = diagnostics_path_from_cfg(
            frame_cfg,
            "log/foot_event_observer_frames.csv");
        cfg.frame_log_enabled =
            frame_cfg && frame_cfg["enabled"]
                ? frame_cfg["enabled"].as<bool>(false)
                : false;
        cfg.flush_interval_events =
            std::max(1, observer_cfg["flush_interval_events"].as<int>(10));
        cfg.flush_interval_frames =
            std::max(
                1,
                frame_cfg && frame_cfg["flush_interval_steps"]
                    ? frame_cfg["flush_interval_steps"].as<int>(50)
                    : 50);
        cfg.feature_dim =
            std::max(1, observer_cfg["feature_dim"].as<int>(93));
        cfg.history_frames =
            std::max(0, observer_cfg["history_frames"].as<int>(0));
        cfg.history_oldest_first =
            observer_cfg["history_oldest_first"].as<bool>(true);
        cfg.enabled = observer_cfg["enabled"].as<bool>(false);

        const std::string common_input_name =
            observer_cfg["input_name"].as<std::string>("obs_history");
        const std::string common_output_name =
            observer_cfg["output_name"].as<std::string>("event_logits");
        const std::string common_activation =
            observer_cfg["activation"].as<std::string>("sigmoid");
        const auto footprint_cfg =
            observer_cfg["footprint_touchdown_detector"]
                ? observer_cfg["footprint_touchdown_detector"]
                : YAML::Node{};
        const auto toe_cfg =
            observer_cfg["toe_riser_detector"]
                ? observer_cfg["toe_riser_detector"]
                : YAML::Node{};

        cfg.footprint_touchdown_detector.name =
            footprint_cfg["name"].as<std::string>("footprint_touchdown_detector");
        cfg.footprint_touchdown_detector.model_path = model_path_from_cfg(
            footprint_cfg,
            "robots/g1/config/policy/velocity/slow_latent/exported/"
            "footprint_touchdown_detector.onnx");
        cfg.footprint_touchdown_detector.input_name =
            footprint_cfg["input_name"].as<std::string>(common_input_name);
        cfg.footprint_touchdown_detector.output_name =
            footprint_cfg["output_name"].as<std::string>(common_output_name);
        cfg.footprint_touchdown_detector.activation =
            footprint_cfg["activation"].as<std::string>(common_activation);

        cfg.toe_riser_detector.name =
            toe_cfg["name"].as<std::string>("toe_riser_detector");
        cfg.toe_riser_detector.model_path = model_path_from_cfg(
            toe_cfg,
            "robots/g1/config/policy/velocity/slow_latent/exported/"
            "toe_riser_detector.onnx");
        cfg.toe_riser_detector.input_name =
            toe_cfg["input_name"].as<std::string>(common_input_name);
        cfg.toe_riser_detector.output_name =
            toe_cfg["output_name"].as<std::string>(common_output_name);
        cfg.toe_riser_detector.activation =
            toe_cfg["activation"].as<std::string>(common_activation);

        cfg.left_touchdown_index =
            footprint_cfg["left_touchdown_index"].as<int>(
                observer_cfg["left_touchdown_index"].as<int>(2));
        cfg.right_touchdown_index =
            footprint_cfg["right_touchdown_index"].as<int>(
                observer_cfg["right_touchdown_index"].as<int>(3));
        cfg.left_contact_index =
            footprint_cfg["left_contact_index"].as<int>(
                observer_cfg["left_contact_index"].as<int>(0));
        cfg.right_contact_index =
            footprint_cfg["right_contact_index"].as<int>(
                observer_cfg["right_contact_index"].as<int>(1));
        cfg.left_toe_riser_index =
            toe_cfg["left_toe_riser_index"].as<int>(
                observer_cfg["left_toe_riser_index"].as<int>(4));
        cfg.right_toe_riser_index =
            toe_cfg["right_toe_riser_index"].as<int>(
                observer_cfg["right_toe_riser_index"].as<int>(5));
        cfg.touchdown_threshold =
            observer_cfg["touchdown_threshold"].as<float>(0.5f);
        cfg.toe_riser_threshold =
            observer_cfg["toe_riser_threshold"].as<float>(0.5f);
        cfg.left_touchdown_threshold =
            footprint_cfg["left_touchdown_threshold"].as<float>(
                observer_cfg["left_touchdown_threshold"].as<float>(-1.0f));
        cfg.right_touchdown_threshold =
            footprint_cfg["right_touchdown_threshold"].as<float>(
                observer_cfg["right_touchdown_threshold"].as<float>(-1.0f));
        cfg.left_toe_riser_threshold =
            toe_cfg["left_toe_riser_threshold"].as<float>(
                observer_cfg["left_toe_riser_threshold"].as<float>(-1.0f));
        cfg.right_toe_riser_threshold =
            toe_cfg["right_toe_riser_threshold"].as<float>(
                observer_cfg["right_toe_riser_threshold"].as<float>(-1.0f));
        cfg.reset_ratio = observer_cfg["reset_ratio"].as<float>(0.75f);
        cfg.touchdown_confirm_frames =
            std::max(1, observer_cfg["touchdown_confirm_frames"].as<int>(1));
        cfg.toe_riser_confirm_frames =
            std::max(1, observer_cfg["toe_riser_confirm_frames"].as<int>(1));
        cfg.touchdown_cooldown_frames =
            std::max(1, observer_cfg["touchdown_cooldown_frames"].as<int>(6));
        cfg.toe_riser_cooldown_frames =
            std::max(1, observer_cfg["toe_riser_cooldown_frames"].as<int>(6));
        cfg.startup_ignore_s =
            observer_cfg["startup_ignore_s"].as<double>(0.5);

        foot_event_observer_.configure(cfg);
        if (!cfg.enabled) {
            return;
        }
        if (foot_event_observer_.open()) {
            spdlog::info(
                "Foot event ONNX observer enabled: footprint_model={} toe_model={} events={}",
                cfg.footprint_touchdown_detector.model_path.string(),
                cfg.toe_riser_detector.model_path.string(),
                cfg.event_path.string());
        } else {
            spdlog::error(
                "Failed to open foot event ONNX observer: footprint_model={} toe_model={} events={}",
                cfg.footprint_touchdown_detector.model_path.string(),
                cfg.toe_riser_detector.model_path.string(),
                cfg.event_path.string());
            throw std::runtime_error(
                "Required foot event ONNX observer failed to start.");
        }
    }

    void configure_foot_event_memory(const YAML::Node& memory_cfg)
    {
        mjlab::diagnostics::FootEventMemoryConfig cfg;
        foot_event_memory_enabled_ = cfg_bool(memory_cfg, "enabled", true);
        cfg.memory_len = cfg_int(memory_cfg, "memory_len", cfg.memory_len);
        cfg.age_norm_s = cfg_float(memory_cfg, "age_norm_s", cfg.age_norm_s);
        cfg.stance_age_norm_s =
            cfg_float(memory_cfg, "stance_age_norm_s", cfg.stance_age_norm_s);
        cfg.release_contact_prob_threshold = cfg_float(
            memory_cfg,
            "release_contact_prob_threshold",
            cfg.release_contact_prob_threshold);
        cfg.release_confirm_frames = cfg_int(
            memory_cfg,
            "release_confirm_frames",
            cfg.release_confirm_frames);
        cfg.early_contact_time_s =
            cfg_float(memory_cfg, "early_contact_time_s", cfg.early_contact_time_s);

        cfg.ratchet_height_threshold_m = cfg_float(
            memory_cfg, "ratchet_height_threshold_m", cfg.ratchet_height_threshold_m);
        cfg.ratchet_flat_height_threshold_m = cfg_float(
            memory_cfg,
            "ratchet_flat_height_threshold_m",
            cfg.ratchet_flat_height_threshold_m);
        cfg.ratchet_probe_increment_m = cfg_float(
            memory_cfg, "ratchet_probe_increment_m", cfg.ratchet_probe_increment_m);
        cfg.ratchet_probe_bootstrap_increment_m = cfg_float(
            memory_cfg,
            "ratchet_probe_bootstrap_increment_m",
            cfg.ratchet_probe_bootstrap_increment_m);
        cfg.ratchet_probe_provisional_min_target_m = cfg_float(
            memory_cfg,
            "ratchet_probe_provisional_min_target_m",
            cfg.ratchet_probe_provisional_min_target_m);
        cfg.ratchet_probe_cap_m =
            cfg_float(memory_cfg, "ratchet_probe_cap_m", cfg.ratchet_probe_cap_m);
        cfg.ratchet_probe_target_tolerance_m = cfg_float(
            memory_cfg,
            "ratchet_probe_target_tolerance_m",
            cfg.ratchet_probe_target_tolerance_m);
        cfg.ratchet_sequence_min_confidence = cfg_float(
            memory_cfg,
            "ratchet_sequence_min_confidence",
            cfg.ratchet_sequence_min_confidence);
        cfg.ratchet_sequence_height_tolerance_m = cfg_float(
            memory_cfg,
            "ratchet_sequence_height_tolerance_m",
            cfg.ratchet_sequence_height_tolerance_m);
        cfg.ratchet_sequence_max_adjacent_height_m = cfg_float(
            memory_cfg,
            "ratchet_sequence_max_adjacent_height_m",
            cfg.ratchet_sequence_max_adjacent_height_m);
        cfg.ratchet_post_first_collision_probe_increment_m = cfg_float(
            memory_cfg,
            "ratchet_post_first_collision_probe_increment_m",
            cfg.ratchet_post_first_collision_probe_increment_m);
        cfg.ratchet_no_hit_lower_margin_m = cfg_float(
            memory_cfg,
            "ratchet_no_hit_lower_margin_m",
            cfg.ratchet_no_hit_lower_margin_m);
        cfg.ratchet_interval_target_margin_m = cfg_float(
            memory_cfg,
            "ratchet_interval_target_margin_m",
            cfg.ratchet_interval_target_margin_m);
        cfg.ratchet_collision_margin_m = cfg_float(
            memory_cfg, "ratchet_collision_margin_m", cfg.ratchet_collision_margin_m);
        cfg.ratchet_toe_anchor_offset_m = cfg_float(
            memory_cfg,
            "ratchet_toe_anchor_offset_m",
            cfg.ratchet_toe_anchor_offset_m);
        cfg.ratchet_recovery_completion_tolerance_m = cfg_float(
            memory_cfg,
            "ratchet_recovery_completion_tolerance_m",
            cfg.ratchet_recovery_completion_tolerance_m);
        cfg.ratchet_recovery_completion_lower_tolerance_m = cfg_float(
            memory_cfg,
            "ratchet_recovery_completion_lower_tolerance_m",
            cfg.ratchet_recovery_completion_lower_tolerance_m);
        cfg.ratchet_recovery_min_reduction_m = cfg_float(
            memory_cfg,
            "ratchet_recovery_min_reduction_m",
            cfg.ratchet_recovery_min_reduction_m);
        cfg.ratchet_recovery_offset_m = cfg_float(
            memory_cfg, "ratchet_recovery_offset_m", cfg.ratchet_recovery_offset_m);
        cfg.ratchet_lock_probe_lower_margin_m = cfg_float(
            memory_cfg,
            "ratchet_lock_probe_lower_margin_m",
            cfg.ratchet_lock_probe_lower_margin_m);
        cfg.ratchet_lock_margin_m =
            cfg_float(memory_cfg, "ratchet_lock_margin_m", cfg.ratchet_lock_margin_m);
        cfg.ratchet_lock_intent_tolerance_m = cfg_float(
            memory_cfg,
            "ratchet_lock_intent_tolerance_m",
            cfg.ratchet_lock_intent_tolerance_m);
        cfg.ratchet_lock_stable_steps = cfg_int(
            memory_cfg, "ratchet_lock_stable_steps", cfg.ratchet_lock_stable_steps);
        cfg.ratchet_lock_target_stable_enabled = cfg_bool(
            memory_cfg,
            "ratchet_lock_target_stable_enabled",
            cfg.ratchet_lock_target_stable_enabled);
        cfg.ratchet_lock_phase_correction_enabled = cfg_bool(
            memory_cfg,
            "ratchet_lock_phase_correction_enabled",
            cfg.ratchet_lock_phase_correction_enabled);
        cfg.ratchet_lock_phase_correction_gain = cfg_float(
            memory_cfg,
            "ratchet_lock_phase_correction_gain",
            cfg.ratchet_lock_phase_correction_gain);
        cfg.ratchet_lock_phase_deadband_m = cfg_float(
            memory_cfg,
            "ratchet_lock_phase_deadband_m",
            cfg.ratchet_lock_phase_deadband_m);
        cfg.ratchet_lock_phase_initial_front_error_m = cfg_float(
            memory_cfg,
            "ratchet_lock_phase_initial_front_error_m",
            cfg.ratchet_lock_phase_initial_front_error_m);
        cfg.ratchet_lock_phase_max_backoff_m = cfg_float(
            memory_cfg,
            "ratchet_lock_phase_max_backoff_m",
            cfg.ratchet_lock_phase_max_backoff_m);
        cfg.ratchet_lock_phase_max_forward_m = cfg_float(
            memory_cfg,
            "ratchet_lock_phase_max_forward_m",
            cfg.ratchet_lock_phase_max_forward_m);
        cfg.ratchet_lock_phase_max_error_m = cfg_float(
            memory_cfg,
            "ratchet_lock_phase_max_error_m",
            cfg.ratchet_lock_phase_max_error_m);
        cfg.ratchet_soft_upper_ttl_steps = cfg_int(
            memory_cfg,
            "ratchet_soft_upper_ttl_steps",
            cfg.ratchet_soft_upper_ttl_steps);
        cfg.ratchet_first_collision_enters_stair_mode = cfg_bool(
            memory_cfg,
            "ratchet_first_collision_enters_stair_mode",
            cfg.ratchet_first_collision_enters_stair_mode);
        cfg.ratchet_single_collision_confirms_interval = cfg_bool(
            memory_cfg,
            "ratchet_single_collision_confirms_interval",
            cfg.ratchet_single_collision_confirms_interval);
        cfg.ratchet_two_collision_enabled = cfg_bool(
            memory_cfg,
            "ratchet_two_collision_enabled",
            cfg.ratchet_two_collision_enabled);
        cfg.ratchet_two_collision_interval_margin_m = cfg_float(
            memory_cfg,
            "ratchet_two_collision_interval_margin_m",
            cfg.ratchet_two_collision_interval_margin_m);
        cfg.ratchet_two_collision_stride_layers = cfg_float(
            memory_cfg,
            "ratchet_two_collision_stride_layers",
            cfg.ratchet_two_collision_stride_layers);
        cfg.ratchet_two_collision_min_layer_delta = cfg_int(
            memory_cfg,
            "ratchet_two_collision_min_layer_delta",
            cfg.ratchet_two_collision_min_layer_delta);
        cfg.ratchet_two_collision_min_height_delta_m = cfg_float(
            memory_cfg,
            "ratchet_two_collision_min_height_delta_m",
            cfg.ratchet_two_collision_min_height_delta_m);
        cfg.ratchet_two_collision_nominal_riser_height_m = cfg_float(
            memory_cfg,
            "ratchet_two_collision_nominal_riser_height_m",
            cfg.ratchet_two_collision_nominal_riser_height_m);
        cfg.ratchet_two_collision_height_layer_tolerance_m = cfg_float(
            memory_cfg,
            "ratchet_two_collision_height_layer_tolerance_m",
            cfg.ratchet_two_collision_height_layer_tolerance_m);
        cfg.ratchet_two_collision_use_height_layers = cfg_bool(
            memory_cfg,
            "ratchet_two_collision_use_height_layers",
            cfg.ratchet_two_collision_use_height_layers);
        cfg.ratchet_two_collision_lower_cross_margin_m = cfg_float(
            memory_cfg,
            "ratchet_two_collision_lower_cross_margin_m",
            cfg.ratchet_two_collision_lower_cross_margin_m);
        cfg.ratchet_two_collision_upper_cross_margin_m = cfg_float(
            memory_cfg,
            "ratchet_two_collision_upper_cross_margin_m",
            cfg.ratchet_two_collision_upper_cross_margin_m);
        cfg.ratchet_rejected_second_hit_confirm_count = cfg_int(
            memory_cfg,
            "ratchet_rejected_second_hit_confirm_count",
            cfg.ratchet_rejected_second_hit_confirm_count);
        cfg.ratchet_rejected_second_hit_target_tolerance_m = cfg_float(
            memory_cfg,
            "ratchet_rejected_second_hit_target_tolerance_m",
            cfg.ratchet_rejected_second_hit_target_tolerance_m);
        cfg.ratchet_second_collision_requires_up_step = cfg_bool(
            memory_cfg,
            "ratchet_second_collision_requires_up_step",
            cfg.ratchet_second_collision_requires_up_step);
        cfg.ratchet_two_collision_tread_min_m = cfg_float(
            memory_cfg,
            "ratchet_two_collision_tread_min_m",
            cfg.ratchet_two_collision_tread_min_m);
        cfg.ratchet_two_collision_tread_max_m = cfg_float(
            memory_cfg,
            "ratchet_two_collision_tread_max_m",
            cfg.ratchet_two_collision_tread_max_m);
        cfg.ratchet_anchor_collision_enabled = cfg_bool(
            memory_cfg,
            "ratchet_anchor_collision_enabled",
            cfg.ratchet_anchor_collision_enabled);
        cfg.ratchet_lower_target_lag_margin_m = cfg_float(
            memory_cfg,
            "ratchet_lower_target_lag_margin_m",
            cfg.ratchet_lower_target_lag_margin_m);
        cfg.ratchet_same_foot_stride_guard_layers = cfg_float(
            memory_cfg,
            "ratchet_same_foot_stride_guard_layers",
            cfg.ratchet_same_foot_stride_guard_layers);
        cfg.ratchet_same_foot_stride_guard_margin_m = cfg_float(
            memory_cfg,
            "ratchet_same_foot_stride_guard_margin_m",
            cfg.ratchet_same_foot_stride_guard_margin_m);
        cfg.ratchet_collision_min_confidence = cfg_float(
            memory_cfg,
            "ratchet_collision_min_confidence",
            cfg.ratchet_collision_min_confidence);
        cfg.ratchet_post_first_collision_collision_min_confidence = cfg_float(
            memory_cfg,
            "ratchet_post_first_collision_collision_min_confidence",
            cfg.ratchet_post_first_collision_collision_min_confidence);
        cfg.ratchet_min_interval_width_m = cfg_float(
            memory_cfg,
            "ratchet_min_interval_width_m",
            cfg.ratchet_min_interval_width_m);
        cfg.ratchet_min_stride_m =
            cfg_float(memory_cfg, "ratchet_min_stride_m", cfg.ratchet_min_stride_m);
        cfg.ratchet_max_stride_m =
            cfg_float(memory_cfg, "ratchet_max_stride_m", cfg.ratchet_max_stride_m);
        cfg.ratchet_reset_flat_pairs = cfg_int(
            memory_cfg, "ratchet_reset_flat_pairs", cfg.ratchet_reset_flat_pairs);

        cfg.odom_contact_lock_threshold = cfg_float(
            memory_cfg,
            "odom_contact_lock_threshold",
            cfg.odom_contact_lock_threshold);
        cfg.odom_contact_release_threshold = cfg_float(
            memory_cfg,
            "odom_contact_release_threshold",
            cfg.odom_contact_release_threshold);
        cfg.odom_max_step_translation_m = cfg_float(
            memory_cfg,
            "odom_max_step_translation_m",
            cfg.odom_max_step_translation_m);
        cfg.odom_max_double_support_residual_m = cfg_float(
            memory_cfg,
            "odom_max_double_support_residual_m",
            cfg.odom_max_double_support_residual_m);

        foot_event_memory_.configure(cfg);
        env->robot->data.foot_event_summary.assign(
            mjlab::diagnostics::FootEventMemory::kSummaryDim,
            0.0f);
    }

    void close_touchdown_log()
    {
        if (touchdown_log_.is_open()) {
            touchdown_log_.flush();
            touchdown_log_.close();
            spdlog::info("Touchdown diagnostics log closed: {}", touchdown_log_path_);
        }
        touchdown_log_enabled_ = false;
    }

    void close_touchdown_event_log()
    {
        touchdown_detector_.close();
    }

    void close_foot_event_observer()
    {
        foot_event_observer_.close();
    }

    void write_touchdown_log_header()
    {
        touchdown_log_
            << "state,step,time_s,lowstate_tick_ms,phase"
            << ",left_expected_stance,right_expected_stance"
            << ",cmd_x,cmd_y,cmd_yaw,stick_ly,stick_lx,stick_rx";
        touchdown_log_ << ",root_quat_w,root_quat_x,root_quat_y,root_quat_z";
        append_vec3_header(touchdown_log_, "gyro_b");
        append_vec3_header(touchdown_log_, "imu_acc_b");
        append_vec3_header(touchdown_log_, "projected_gravity_b");
        append_vec3_header(touchdown_log_, "left_toe_pos_b");
        append_vec3_header(touchdown_log_, "right_toe_pos_b");
        append_vec3_header(touchdown_log_, "left_heel_pos_b");
        append_vec3_header(touchdown_log_, "right_heel_pos_b");
        append_vec3_header(touchdown_log_, "left_toe_vel_b");
        append_vec3_header(touchdown_log_, "right_toe_vel_b");
        append_vec3_header(touchdown_log_, "left_heel_vel_b");
        append_vec3_header(touchdown_log_, "right_heel_vel_b");
        touchdown_log_
            << ",left_tracking_error_norm,right_tracking_error_norm"
            << ",left_tau_abs_mean,right_tau_abs_mean";

        for (int i = 0; i < 29; ++i) {
            const auto name = kJointColumnNames[i];
            touchdown_log_ << ",q_" << name
                           << ",dq_" << name
                           << ",tau_est_" << name
                           << ",raw_action_" << name
                           << ",prev_raw_action_" << name
                           << ",target_q_" << name
                           << ",tracking_error_" << name;
        }
        touchdown_log_ << '\n';
    }

    std::vector<float> current_tau_est() const
    {
        const auto& joint_ids = env->robot->data.joint_ids_map;
        std::vector<float> tau(joint_ids.size(), 0.0f);
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        for (size_t i = 0; i < joint_ids.size(); ++i) {
            tau[i] = lowstate->msg_.motor_state()[joint_ids[i]].tau_est();
        }
        return tau;
    }

    Eigen::Vector3f current_imu_accel() const
    {
        Eigen::Vector3f accel = Eigen::Vector3f::Zero();
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        for (int i = 0; i < 3; ++i) {
            accel[i] = lowstate->msg_.imu_state().accelerometer()[i];
        }
        return accel;
    }

    uint32_t current_lowstate_tick_ms() const
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        return lowstate->msg_.tick();
    }

    mjlab::diagnostics::FootEventObserverFrame make_foot_event_observer_frame(
        const std::string& state_string,
        const uint32_t lowstate_tick_ms,
        const float phase,
        const std::vector<float>& command,
        const std::vector<float>& prev_action,
        const std::vector<float>& scale) const
    {
        const auto& data = env->robot->data;
        mjlab::diagnostics::FootEventObserverFrame frame;
        frame.state = state_string;
        frame.step = env->episode_length;
        frame.tick_ms = lowstate_tick_ms;
        frame.time_s =
            static_cast<double>(env->episode_length) * env->step_dt;
        frame.phase = phase;
        frame.command = {command[0], command[1], command[2]};
        frame.joystick = {
            data.joystick->ly(),
            data.joystick->lx(),
            data.joystick->rx(),
        };
        frame.root_quat_w = data.root_quat_w;
        frame.projected_gravity_b = data.projected_gravity_b;
        frame.root_ang_vel_b = data.root_ang_vel_b;
        frame.left_toe_pos_b = data.left_toe_pos_body;
        frame.right_toe_pos_b = data.right_toe_pos_body;
        frame.left_heel_pos_b = data.left_heel_pos_body;
        frame.right_heel_pos_b = data.right_heel_pos_body;
        frame.prev_action = prev_action;
        frame.action_scale = scale;
        frame.joint_pos = data.joint_pos;
        frame.default_joint_pos = data.default_joint_pos;
        frame.joint_vel = data.joint_vel;
        return frame;
    }

    void clear_foot_event_memory_summary()
    {
        foot_event_memory_.reset();
        env->robot->data.foot_event_summary.assign(
            mjlab::diagnostics::FootEventMemory::kSummaryDim,
            0.0f);
    }

    void reset_foot_event_runtime()
    {
        foot_event_observer_.reset();
        clear_foot_event_memory_summary();
    }

    void update_foot_event_memory_before_policy()
    {
        if (!foot_event_memory_enabled_) {
            clear_foot_event_memory_summary();
            return;
        }
        if (!foot_event_observer_.enabled()) {
            clear_foot_event_memory_summary();
            return;
        }

        const auto command = isaaclab::mdp::velocity_commands(env.get(), YAML::Node{});
        const auto prev_action = env->action_manager->prev_action();
        const auto scale = action_scale();
        const float phase = current_gait_phase();
        const auto observer_frame = make_foot_event_observer_frame(
            getStateString(),
            current_lowstate_tick_ms(),
            phase,
            command,
            prev_action,
            scale);
        const auto observer_output = foot_event_observer_.update(observer_frame);
        if (!foot_event_observer_.enabled()) {
            clear_foot_event_memory_summary();
            return;
        }

        mjlab::diagnostics::FootEventMemoryFrame memory_frame;
        const auto& data = env->robot->data;
        memory_frame.step = env->episode_length;
        memory_frame.dt = env->step_dt;
        memory_frame.phase = phase;
        memory_frame.command = {command[0], command[1], command[2]};
        memory_frame.root_quat_w = data.root_quat_w;
        memory_frame.left_toe_pos_b = data.left_toe_pos_body;
        memory_frame.right_toe_pos_b = data.right_toe_pos_body;
        memory_frame.left_heel_pos_b = data.left_heel_pos_body;
        memory_frame.right_heel_pos_b = data.right_heel_pos_body;
        memory_frame.contact_prob = {
            observer_output.left_contact_prob,
            observer_output.right_contact_prob,
        };
        memory_frame.touchdown_prob = {
            observer_output.left_touchdown_prob,
            observer_output.right_touchdown_prob,
        };
        memory_frame.toe_riser_prob = {
            observer_output.left_toe_riser_prob,
            observer_output.right_toe_riser_prob,
        };
        memory_frame.touchdown_event = {
            observer_output.left_touchdown_event,
            observer_output.right_touchdown_event,
        };
        memory_frame.toe_riser_event = {
            observer_output.left_toe_riser_event,
            observer_output.right_toe_riser_event,
        };
        env->robot->data.foot_event_summary = foot_event_memory_.update(memory_frame);
    }

    void write_touchdown_log_row()
    {
        const bool write_frame_log = touchdown_log_enabled_ && touchdown_log_.is_open();
        if (!write_frame_log &&
            !touchdown_detector_.enabled()) {
            return;
        }

        const auto& data = env->robot->data;
        const std::string state_string = getStateString();
        const uint32_t lowstate_tick_ms = current_lowstate_tick_ms();
        const auto command = isaaclab::mdp::velocity_commands(env.get(), YAML::Node{});
        const auto raw_action = env->action_manager->action();
        const auto prev_action = env->action_manager->prev_action();
        const auto processed_action = env->action_manager->processed_actions();
        const auto scale = action_scale();
        const auto offset = action_offset();
        const auto tau = current_tau_est();
        const auto imu_accel = current_imu_accel();

        Eigen::Vector3f left_toe_vel = Eigen::Vector3f::Zero();
        Eigen::Vector3f right_toe_vel = Eigen::Vector3f::Zero();
        Eigen::Vector3f left_heel_vel = Eigen::Vector3f::Zero();
        Eigen::Vector3f right_heel_vel = Eigen::Vector3f::Zero();
        if (touchdown_log_prev_valid_) {
            left_toe_vel = (data.left_toe_pos_body - prev_left_toe_pos_body_) /
                           env->step_dt;
            right_toe_vel = (data.right_toe_pos_body - prev_right_toe_pos_body_) /
                            env->step_dt;
            left_heel_vel = (data.left_heel_pos_body - prev_left_heel_pos_body_) /
                            env->step_dt;
            right_heel_vel = (data.right_heel_pos_body - prev_right_heel_pos_body_) /
                             env->step_dt;
        }

        prev_left_toe_pos_body_ = data.left_toe_pos_body;
        prev_right_toe_pos_body_ = data.right_toe_pos_body;
        prev_left_heel_pos_body_ = data.left_heel_pos_body;
        prev_right_heel_pos_body_ = data.right_heel_pos_body;
        touchdown_log_prev_valid_ = true;

        const float phase = current_gait_phase();
        const bool left_expected_stance = phase < touchdown_log_stance_threshold_;
        const bool right_expected_stance =
            std::fmod(phase + 0.5f, 1.0f) < touchdown_log_stance_threshold_;

        float left_tracking_norm_sq = 0.0f;
        float right_tracking_norm_sq = 0.0f;
        float left_tau_abs_sum = 0.0f;
        float right_tau_abs_sum = 0.0f;
        for (int i = 0; i < 6; ++i) {
            const float target = i < static_cast<int>(processed_action.size())
                ? processed_action[i]
                : (i < static_cast<int>(offset.size()) ? offset[i] : 0.0f);
            const float error = target - data.joint_pos[i];
            left_tracking_norm_sq += error * error;
            if (i < static_cast<int>(tau.size())) {
                left_tau_abs_sum += std::abs(tau[i]);
            }
        }
        for (int i = 6; i < 12; ++i) {
            const float target = i < static_cast<int>(processed_action.size())
                ? processed_action[i]
                : (i < static_cast<int>(offset.size()) ? offset[i] : 0.0f);
            const float error = target - data.joint_pos[i];
            right_tracking_norm_sq += error * error;
            if (i < static_cast<int>(tau.size())) {
                right_tau_abs_sum += std::abs(tau[i]);
            }
        }

        const float left_tracking_norm = std::sqrt(left_tracking_norm_sq);
        const float right_tracking_norm = std::sqrt(right_tracking_norm_sq);
        const float left_tau_mean = left_tau_abs_sum / 6.0f;
        const float right_tau_mean = right_tau_abs_sum / 6.0f;

        float imu_jerk = 0.0f;
        float left_tracking_delta = 0.0f;
        float right_tracking_delta = 0.0f;
        float left_tau_delta = 0.0f;
        float right_tau_delta = 0.0f;
        if (touchdown_log_prev_imu_valid_) {
            imu_jerk = (imu_accel - prev_imu_accel_).norm() / env->step_dt;
            left_tracking_delta = left_tracking_norm - prev_left_tracking_norm_;
            right_tracking_delta = right_tracking_norm - prev_right_tracking_norm_;
            left_tau_delta = left_tau_mean - prev_left_tau_mean_;
            right_tau_delta = right_tau_mean - prev_right_tau_mean_;
        }

        prev_imu_accel_ = imu_accel;
        prev_left_tracking_norm_ = left_tracking_norm;
        prev_right_tracking_norm_ = right_tracking_norm;
        prev_left_tau_mean_ = left_tau_mean;
        prev_right_tau_mean_ = right_tau_mean;
        touchdown_log_prev_imu_valid_ = true;

        if (touchdown_detector_.enabled()) {
            mjlab::diagnostics::TouchdownDetectorFrame detector_frame;
            detector_frame.state = state_string;
            detector_frame.step = env->episode_length;
            detector_frame.tick_ms = lowstate_tick_ms;
            detector_frame.time_s =
                static_cast<double>(env->episode_length) * env->step_dt;
            detector_frame.phase = phase;
            detector_frame.command = {command[0], command[1], command[2]};
            detector_frame.joystick = {
                data.joystick->ly(),
                data.joystick->lx(),
                data.joystick->rx(),
            };
            detector_frame.imu_jerk = imu_jerk;
            detector_frame.left.expected_stance = left_expected_stance;
            detector_frame.left.toe_pos = data.left_toe_pos_body;
            detector_frame.left.heel_pos = data.left_heel_pos_body;
            detector_frame.left.toe_vel = left_toe_vel;
            detector_frame.left.heel_vel = left_heel_vel;
            detector_frame.left.tracking_error_norm = left_tracking_norm;
            detector_frame.left.tau_abs_mean = left_tau_mean;
            detector_frame.left.tracking_delta = left_tracking_delta;
            detector_frame.left.tau_delta = left_tau_delta;
            detector_frame.right.expected_stance = right_expected_stance;
            detector_frame.right.toe_pos = data.right_toe_pos_body;
            detector_frame.right.heel_pos = data.right_heel_pos_body;
            detector_frame.right.toe_vel = right_toe_vel;
            detector_frame.right.heel_vel = right_heel_vel;
            detector_frame.right.tracking_error_norm = right_tracking_norm;
            detector_frame.right.tau_abs_mean = right_tau_mean;
            detector_frame.right.tracking_delta = right_tracking_delta;
            detector_frame.right.tau_delta = right_tau_delta;
            touchdown_detector_.update(detector_frame);
        }

        if (!write_frame_log) {
            return;
        }

        touchdown_log_
            << state_string
            << ',' << env->episode_length
            << ',' << static_cast<double>(env->episode_length) * env->step_dt
            << ',' << lowstate_tick_ms
            << ',' << phase
            << ',' << static_cast<int>(left_expected_stance)
            << ',' << static_cast<int>(right_expected_stance)
            << ',' << command[0]
            << ',' << command[1]
            << ',' << command[2]
            << ',' << data.joystick->ly()
            << ',' << data.joystick->lx()
            << ',' << data.joystick->rx()
            << ',' << data.root_quat_w.w()
            << ',' << data.root_quat_w.x()
            << ',' << data.root_quat_w.y()
            << ',' << data.root_quat_w.z()
            << ',' << data.root_ang_vel_b.x()
            << ',' << data.root_ang_vel_b.y()
            << ',' << data.root_ang_vel_b.z();
        append_vec3(touchdown_log_, imu_accel);
        append_vec3(touchdown_log_, data.projected_gravity_b);
        append_vec3(touchdown_log_, data.left_toe_pos_body);
        append_vec3(touchdown_log_, data.right_toe_pos_body);
        append_vec3(touchdown_log_, data.left_heel_pos_body);
        append_vec3(touchdown_log_, data.right_heel_pos_body);
        append_vec3(touchdown_log_, left_toe_vel);
        append_vec3(touchdown_log_, right_toe_vel);
        append_vec3(touchdown_log_, left_heel_vel);
        append_vec3(touchdown_log_, right_heel_vel);
        touchdown_log_
            << ',' << left_tracking_norm
            << ',' << right_tracking_norm
            << ',' << left_tau_mean
            << ',' << right_tau_mean;

        for (int i = 0; i < 29; ++i) {
            const float q = i < data.joint_pos.size() ? data.joint_pos[i] : 0.0f;
            const float dq = i < data.joint_vel.size() ? data.joint_vel[i] : 0.0f;
            const float tau_est = i < static_cast<int>(tau.size()) ? tau[i] : 0.0f;
            const float raw = i < static_cast<int>(raw_action.size())
                ? raw_action[i] : 0.0f;
            const float prev_raw = i < static_cast<int>(prev_action.size())
                ? prev_action[i] : 0.0f;
            const float target = i < static_cast<int>(processed_action.size())
                ? processed_action[i]
                : ((i < static_cast<int>(offset.size()) ? offset[i] : 0.0f) +
                   raw * (i < static_cast<int>(scale.size()) ? scale[i] : 1.0f));
            touchdown_log_
                << ',' << q
                << ',' << dq
                << ',' << tau_est
                << ',' << raw
                << ',' << prev_raw
                << ',' << target
                << ',' << target - q;
        }
        touchdown_log_ << '\n';

        if (env->episode_length % touchdown_log_flush_interval_steps_ == 0) {
            touchdown_log_.flush();
        }
    }

    void wait_for_camera_startup()
    {
        const auto camera_cfg_root = env->cfg["camera"];
        if (!camera_cfg_root || !camera_cfg_root["front_depth"]) {
            return;
        }

        const auto camera_cfg = camera_cfg_root["front_depth"];
        if (!camera_cfg["wait_for_first_frame"].as<bool>(false)) {
            return;
        }

        const std::string camera_name = camera_cfg["camera_name"].as<std::string>("front_depth");
        const int width = camera_cfg["width"].as<int>(64);
        const int height = camera_cfg["height"].as<int>(36);
        const int channels = camera_cfg["channels"].as<int>(
            camera_cfg["stack_length"].as<int>(8)
        );
        const size_t expected_size = static_cast<size_t>(std::max(0, width)) *
                                     static_cast<size_t>(std::max(0, height)) *
                                     static_cast<size_t>(std::max(1, channels));
        const float timeout_s = camera_cfg["startup_timeout_s"].as<float>(1.0f);

        using clock = std::chrono::steady_clock;
        const auto deadline = clock::now() + std::chrono::duration<double>(timeout_s);
        while (policy_thread_running && clock::now() < deadline) {
            env->robot->update();
            const auto it = env->robot->data.camera_frames.find(camera_name);
            if (it != env->robot->data.camera_frames.end() &&
                it->second.size() == expected_size) {
                env->reset();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        env->reset();
    }

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::ofstream touchdown_log_;
    std::string touchdown_log_path_;
    std::filesystem::path diagnostics_run_dir_;
    int touchdown_log_flush_interval_steps_ = 50;
    float touchdown_log_stance_threshold_ = 0.56f;
    bool touchdown_log_enabled_ = false;
    bool touchdown_log_prev_valid_ = false;
    bool touchdown_log_prev_imu_valid_ = false;
    Eigen::Vector3f prev_left_toe_pos_body_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_right_toe_pos_body_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_left_heel_pos_body_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_right_heel_pos_body_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_imu_accel_ = Eigen::Vector3f::Zero();
    float prev_left_tracking_norm_ = 0.0f;
    float prev_right_tracking_norm_ = 0.0f;
    float prev_left_tau_mean_ = 0.0f;
    float prev_right_tau_mean_ = 0.0f;
    mjlab::diagnostics::TouchdownDetector touchdown_detector_;
    mjlab::diagnostics::FootEventObserver foot_event_observer_;
    mjlab::diagnostics::FootEventMemory foot_event_memory_;
    bool foot_event_memory_enabled_ = true;

    std::thread policy_thread;
    bool policy_thread_running = false;
    bool entry_posture_active_ = false;
    float entry_posture_duration_s_ = 0.0f;
    std::chrono::steady_clock::time_point entry_posture_start_;
    std::vector<float> entry_posture_q0_;
    std::vector<float> entry_posture_target_;
};

REGISTER_FSM(State_RLBase)
