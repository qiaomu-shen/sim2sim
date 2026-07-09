// Copyright (c) 2026
// Runtime touchdown event detector for deploy-side diagnostics.

#pragma once

#include <eigen3/Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace mjlab::diagnostics
{

struct TouchdownDetectorConfig
{
    bool enabled = false;
    std::filesystem::path path;
    int flush_interval_events = 10;
    int score_threshold = 3;
    int off_confirm_frames = 3;
    int on_confirm_frames = 2;
    int buffer_frames = 16;
    int event_search_frames = 8;
    int pre_window_frames = 3;
    int post_window_frames = 3;
    int post_confirm_frames = 3;
    int min_air_frames = 5;
    int cooldown_frames = 10;
    bool require_motion_cue = true;
    double startup_ignore_s = 0.5;
    float contact_z_threshold = -0.74f;
    float air_z_threshold = -0.70f;
    float liftoff_vz_threshold = 0.05f;
    float vz_brake_down_threshold = -0.02f;
    float vz_brake_abs_threshold = 0.08f;
    float hspeed_threshold = 0.18f;
    float tracking_delta_threshold = 0.03f;
    float tau_delta_threshold = 0.5f;
    float imu_jerk_threshold = 80.0f;
};

struct FootDetectorFrame
{
    bool expected_stance = false;
    Eigen::Vector3f toe_pos = Eigen::Vector3f::Zero();
    Eigen::Vector3f heel_pos = Eigen::Vector3f::Zero();
    Eigen::Vector3f toe_vel = Eigen::Vector3f::Zero();
    Eigen::Vector3f heel_vel = Eigen::Vector3f::Zero();
    float tracking_error_norm = 0.0f;
    float tau_abs_mean = 0.0f;
    float tracking_delta = 0.0f;
    float tau_delta = 0.0f;
};

struct TouchdownDetectorFrame
{
    std::string state;
    int step = 0;
    uint32_t tick_ms = 0;
    double time_s = 0.0;
    float phase = 0.0f;
    std::array<float, 3> command = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> joystick = {0.0f, 0.0f, 0.0f};
    float imu_jerk = 0.0f;
    FootDetectorFrame left;
    FootDetectorFrame right;
};

class TouchdownDetector
{
public:
    void configure(const TouchdownDetectorConfig& config)
    {
        close();
        cfg_ = config;
        reset();
    }

    bool enabled() const
    {
        return cfg_.enabled;
    }

    bool open()
    {
        if (!cfg_.enabled) {
            return false;
        }
        if (cfg_.path.has_parent_path()) {
            std::filesystem::create_directories(cfg_.path.parent_path());
        }
        stream_.open(cfg_.path, std::ios::out | std::ios::trunc);
        if (!stream_) {
            cfg_.enabled = false;
            return false;
        }
        stream_.setf(std::ios::fixed);
        stream_ << std::setprecision(7);
        write_header();
        stream_.flush();
        return true;
    }

    void close()
    {
        if (stream_.is_open()) {
            stream_.flush();
            stream_.close();
        }
    }

    void reset()
    {
        left_ = FootState{};
        right_ = FootState{};
        event_count_ = 0;
    }

    void update(const TouchdownDetectorFrame& frame)
    {
        if (!cfg_.enabled || !stream_.is_open()) {
            return;
        }
        update_foot("left", frame.left, frame, left_);
        update_foot("right", frame.right, frame, right_);
    }

private:
    struct ScoreComponents
    {
        bool expected_stance = false;
        bool low_foot_z = false;
        bool vz_brakes = false;
        bool horizontal_slows = false;
        bool tracking_rises = false;
        bool tau_rises = false;
        bool imu_impulse = false;
    };

    struct FootSample
    {
        std::string state;
        int step = 0;
        uint32_t tick_ms = 0;
        double time_s = 0.0;
        float phase = 0.0f;
        std::array<float, 3> command = {0.0f, 0.0f, 0.0f};
        std::array<float, 3> joystick = {0.0f, 0.0f, 0.0f};
        Eigen::Vector3f toe_pos = Eigen::Vector3f::Zero();
        Eigen::Vector3f heel_pos = Eigen::Vector3f::Zero();
        Eigen::Vector3f foot_pos = Eigen::Vector3f::Zero();
        float foot_z = 0.0f;
        float foot_vz = 0.0f;
        float hspeed = 0.0f;
        float tracking_error_norm = 0.0f;
        float tau_abs_mean = 0.0f;
        float tracking_delta = 0.0f;
        float tau_delta = 0.0f;
        float imu_jerk = 0.0f;
        bool airborne = false;
        int score = 0;
        ScoreComponents components;
    };

    struct PendingEvent
    {
        std::string foot;
        FootSample event_sample;
        FootSample confirm_sample;
        int due_step = 0;
    };

    struct FootState
    {
        enum class Mode
        {
            Contact,
            Air
        };

        Mode mode = Mode::Contact;
        bool has_seen_air = false;
        int air_frames = 0;
        int high_score_frames = 0;
        int last_liftoff_step = -1000000;
        int last_touchdown_step = -1000000;
        std::deque<FootSample> history;
        std::deque<PendingEvent> pending;
    };

    static void append_vec3(std::ofstream& stream, const Eigen::Vector3f& value)
    {
        stream << ',' << value.x()
               << ',' << value.y()
               << ',' << value.z();
    }

    void write_header()
    {
        stream_
            << "event_type,foot,event_step,event_tick_ms,event_time_s"
            << ",confirm_step,confirm_tick_ms,confirm_time_s,confirm_delay_ms"
            << ",state,phase,score,confidence,cmd_x,cmd_y,cmd_yaw"
            << ",stick_ly,stick_lx,stick_rx"
            << ",foot_pos_b_x,foot_pos_b_y,foot_pos_b_z"
            << ",toe_pos_b_x,toe_pos_b_y,toe_pos_b_z"
            << ",heel_pos_b_x,heel_pos_b_y,heel_pos_b_z"
            << ",pre_vz,post_vz,pre_hspeed,post_hspeed"
            << ",tracking_error_norm,tau_abs_mean,tracking_delta,tau_delta,imu_jerk"
            << ",expected_stance,low_foot_z,vz_brakes,horizontal_slows"
            << ",tracking_rises,tau_rises,imu_impulse,airborne\n";
    }

    FootSample make_sample(
        const FootDetectorFrame& foot_frame,
        const TouchdownDetectorFrame& frame,
        const FootState& state) const
    {
        FootSample sample;
        sample.state = frame.state;
        sample.step = frame.step;
        sample.tick_ms = frame.tick_ms;
        sample.time_s = frame.time_s;
        sample.phase = frame.phase;
        sample.command = frame.command;
        sample.joystick = frame.joystick;
        sample.toe_pos = foot_frame.toe_pos;
        sample.heel_pos = foot_frame.heel_pos;
        sample.foot_pos = 0.5f * (foot_frame.toe_pos + foot_frame.heel_pos);
        sample.foot_z = sample.foot_pos.z();

        const Eigen::Vector3f foot_vel =
            0.5f * (foot_frame.toe_vel + foot_frame.heel_vel);
        sample.foot_vz = foot_vel.z();
        sample.hspeed =
            0.5f * (
                std::hypot(foot_frame.toe_vel.x(), foot_frame.toe_vel.y()) +
                std::hypot(foot_frame.heel_vel.x(), foot_frame.heel_vel.y())
            );
        sample.tracking_error_norm = foot_frame.tracking_error_norm;
        sample.tau_abs_mean = foot_frame.tau_abs_mean;
        sample.tracking_delta = foot_frame.tracking_delta;
        sample.tau_delta = foot_frame.tau_delta;
        sample.imu_jerk = frame.imu_jerk;
        sample.airborne =
            sample.foot_z > cfg_.air_z_threshold ||
            (sample.foot_z > cfg_.contact_z_threshold &&
             sample.foot_vz > cfg_.liftoff_vz_threshold);

        float recent_min_vz = sample.foot_vz;
        float recent_hspeed = sample.hspeed;
        int recent_count = 1;
        for (auto it = state.history.rbegin();
             it != state.history.rend() && recent_count < cfg_.pre_window_frames + 1;
             ++it, ++recent_count) {
            recent_min_vz = std::min(recent_min_vz, it->foot_vz);
            recent_hspeed += it->hspeed;
        }
        recent_hspeed /= static_cast<float>(std::max(1, recent_count));

        sample.components.expected_stance = foot_frame.expected_stance;
        sample.components.low_foot_z = sample.foot_z <= cfg_.contact_z_threshold;
        sample.components.vz_brakes =
            recent_min_vz < cfg_.vz_brake_down_threshold &&
            std::abs(sample.foot_vz) < cfg_.vz_brake_abs_threshold;
        sample.components.horizontal_slows =
            sample.hspeed <= cfg_.hspeed_threshold ||
            sample.hspeed < 0.85f * recent_hspeed;
        sample.components.tracking_rises =
            foot_frame.tracking_delta > cfg_.tracking_delta_threshold;
        sample.components.tau_rises =
            foot_frame.tau_delta > cfg_.tau_delta_threshold;
        sample.components.imu_impulse = frame.imu_jerk > cfg_.imu_jerk_threshold;

        sample.score =
            static_cast<int>(sample.components.expected_stance) +
            static_cast<int>(sample.components.low_foot_z) +
            static_cast<int>(sample.components.vz_brakes) +
            static_cast<int>(sample.components.horizontal_slows) +
            static_cast<int>(sample.components.tracking_rises) +
            static_cast<int>(sample.components.tau_rises) +
            static_cast<int>(sample.components.imu_impulse);
        return sample;
    }

    FootSample select_event_sample(const FootState& state, const FootSample& confirm)
    {
        const int earliest_step = confirm.step - cfg_.event_search_frames;
        for (const auto& sample : state.history) {
            if (sample.step >= earliest_step &&
                sample.score >= cfg_.score_threshold &&
                has_required_touchdown_cue(sample)) {
                return sample;
            }
        }

        FootSample best = confirm;
        for (const auto& sample : state.history) {
            if (sample.step >= earliest_step && sample.foot_z < best.foot_z) {
                best = sample;
            }
        }
        return best;
    }

    bool has_required_touchdown_cue(const FootSample& sample) const
    {
        if (!cfg_.require_motion_cue) {
            return true;
        }
        return sample.components.vz_brakes ||
               sample.components.horizontal_slows ||
               sample.components.imu_impulse;
    }

    void update_foot(
        const std::string& foot,
        const FootDetectorFrame& foot_frame,
        const TouchdownDetectorFrame& frame,
        FootState& state)
    {
        FootSample sample = make_sample(foot_frame, frame, state);
        state.history.push_back(sample);
        while (static_cast<int>(state.history.size()) > cfg_.buffer_frames) {
            state.history.pop_front();
        }

        flush_ready_events(state, frame.step);

        if (frame.time_s < cfg_.startup_ignore_s) {
            return;
        }

        if (sample.airborne) {
            state.air_frames += 1;
        } else {
            state.air_frames = 0;
        }

        if (state.mode == FootState::Mode::Contact) {
            state.high_score_frames = 0;
            if (state.air_frames >= cfg_.off_confirm_frames) {
                state.mode = FootState::Mode::Air;
                state.has_seen_air = true;
                state.last_liftoff_step = sample.step;
            }
            return;
        }

        const bool cue_ok = has_required_touchdown_cue(sample);
        if (!sample.airborne && cue_ok && sample.score >= cfg_.score_threshold &&
            sample.step - state.last_touchdown_step >= cfg_.cooldown_frames) {
            state.high_score_frames += 1;
        } else if (sample.airborne || !cue_ok ||
                   sample.score < cfg_.score_threshold) {
            state.high_score_frames = 0;
        }

        const bool has_enough_air =
            sample.step - state.last_liftoff_step >= cfg_.min_air_frames;
        if (state.has_seen_air &&
            has_enough_air &&
            state.high_score_frames >= cfg_.on_confirm_frames) {
            const FootSample event_sample = select_event_sample(state, sample);
            state.pending.push_back(PendingEvent{
                foot,
                event_sample,
                sample,
                sample.step + cfg_.post_confirm_frames
            });
            state.mode = FootState::Mode::Contact;
            state.last_touchdown_step = event_sample.step;
            state.high_score_frames = 0;
            state.air_frames = 0;
        }
    }

    float mean_window(
        const std::deque<FootSample>& history,
        int begin_step,
        int end_step,
        float FootSample::* field) const
    {
        float sum = 0.0f;
        int count = 0;
        for (const auto& sample : history) {
            if (sample.step >= begin_step && sample.step < end_step) {
                sum += sample.*field;
                count += 1;
            }
        }
        return count > 0 ? sum / static_cast<float>(count) : 0.0f;
    }

    void flush_ready_events(FootState& state, int current_step)
    {
        while (!state.pending.empty() && state.pending.front().due_step <= current_step) {
            write_event(state.pending.front(), state.history);
            state.pending.pop_front();
        }
    }

    void write_event(
        const PendingEvent& event,
        const std::deque<FootSample>& history)
    {
        const auto& sample = event.event_sample;
        const auto& confirm = event.confirm_sample;
        const int pre_begin = sample.step - cfg_.pre_window_frames;
        const int pre_end = sample.step;
        const int post_begin = sample.step + 1;
        const int post_end = sample.step + 1 + cfg_.post_window_frames;
        const float pre_vz = mean_window(history, pre_begin, pre_end, &FootSample::foot_vz);
        const float post_vz = mean_window(history, post_begin, post_end, &FootSample::foot_vz);
        const float pre_hspeed =
            mean_window(history, pre_begin, pre_end, &FootSample::hspeed);
        const float post_hspeed =
            mean_window(history, post_begin, post_end, &FootSample::hspeed);
        const double confirm_delay_ms =
            confirm.tick_ms >= sample.tick_ms && sample.tick_ms > 0
                ? static_cast<double>(confirm.tick_ms - sample.tick_ms)
                : (confirm.time_s - sample.time_s) * 1000.0;

        const char* confidence =
            sample.score >= cfg_.score_threshold + 2 ? "high" :
            sample.score >= cfg_.score_threshold + 1 ? "medium" : "low";

        stream_ << "detector_touchdown"
                << ',' << event.foot
                << ',' << sample.step
                << ',' << sample.tick_ms
                << ',' << sample.time_s
                << ',' << confirm.step
                << ',' << confirm.tick_ms
                << ',' << confirm.time_s
                << ',' << confirm_delay_ms
                << ',' << sample.state
                << ',' << sample.phase
                << ',' << sample.score
                << ',' << confidence
                << ',' << sample.command[0]
                << ',' << sample.command[1]
                << ',' << sample.command[2]
                << ',' << sample.joystick[0]
                << ',' << sample.joystick[1]
                << ',' << sample.joystick[2];
        append_vec3(stream_, sample.foot_pos);
        append_vec3(stream_, sample.toe_pos);
        append_vec3(stream_, sample.heel_pos);
        stream_ << ',' << pre_vz
                << ',' << post_vz
                << ',' << pre_hspeed
                << ',' << post_hspeed
                << ',' << sample.tracking_error_norm
                << ',' << sample.tau_abs_mean
                << ',' << sample.tracking_delta
                << ',' << sample.tau_delta
                << ',' << sample.imu_jerk
                << ',' << static_cast<int>(sample.components.expected_stance)
                << ',' << static_cast<int>(sample.components.low_foot_z)
                << ',' << static_cast<int>(sample.components.vz_brakes)
                << ',' << static_cast<int>(sample.components.horizontal_slows)
                << ',' << static_cast<int>(sample.components.tracking_rises)
                << ',' << static_cast<int>(sample.components.tau_rises)
                << ',' << static_cast<int>(sample.components.imu_impulse)
                << ',' << static_cast<int>(sample.airborne)
                << '\n';

        event_count_ += 1;
        if (event_count_ % cfg_.flush_interval_events == 0) {
            stream_.flush();
        }
    }

    TouchdownDetectorConfig cfg_;
    FootState left_;
    FootState right_;
    std::ofstream stream_;
    int event_count_ = 0;
};

}  // namespace mjlab::diagnostics
