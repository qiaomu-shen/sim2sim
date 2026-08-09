// Copyright (c) 2026
// Deploy-side FootEventMemory summary for SlowLatent policies.

#pragma once

#include <eigen3/Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace mjlab::diagnostics
{

struct FootEventMemoryConfig
{
    int memory_len = 6;
    float age_norm_s = 1.5f;
    float stance_age_norm_s = 1.5f;
    float release_contact_prob_threshold = 0.20f;
    int release_confirm_frames = 2;
    float early_contact_time_s = 0.08f;
    bool predicted_fill_enabled = false;
    int predicted_fill_grace_frames = 3;
    float predicted_fill_phase_window = 0.18f;
    float predicted_fill_phase_lead_window = 0.08f;
    float predicted_fill_contact_threshold = 0.35f;
    float predicted_fill_min_command_norm = 0.10f;
    float predicted_fill_confidence = 0.35f;
    float predicted_fill_touchdown_prob = 0.35f;
    bool touchdown_gate_enabled = true;
    float touchdown_gate_contact_threshold = 0.20f;

    float ratchet_height_threshold_m = 0.025f;
    float ratchet_flat_height_threshold_m = 0.02f;
    float ratchet_probe_increment_m = 0.05f;
    float ratchet_probe_bootstrap_increment_m = 0.10f;
    float ratchet_probe_provisional_min_target_m = 0.50f;
    float ratchet_probe_cap_m = 0.76f;
    float ratchet_probe_target_tolerance_m = 0.025f;
    float ratchet_sequence_min_confidence = 0.30f;
    float ratchet_sequence_height_tolerance_m = 0.06f;
    float ratchet_sequence_max_adjacent_height_m = 0.24f;
    float ratchet_no_hit_lower_margin_m = 0.0f;
    float ratchet_interval_target_margin_m = 0.01f;
    float ratchet_collision_margin_m = 0.02f;
    float ratchet_toe_anchor_offset_m = 0.085f;
    float ratchet_recovery_completion_tolerance_m = 0.025f;
    float ratchet_recovery_completion_lower_tolerance_m = 0.025f;
    float ratchet_recovery_min_reduction_m = 0.025f;
    float ratchet_recovery_offset_m = 0.04f;
    float ratchet_lock_probe_lower_margin_m = 0.01f;
    float ratchet_lock_margin_m = 0.005f;
    float ratchet_lock_intent_tolerance_m = 0.03f;
    int ratchet_lock_stable_steps = 2;
    bool ratchet_lock_target_stable_enabled = false;
    bool ratchet_lock_phase_correction_enabled = false;
    float ratchet_lock_phase_correction_gain = 0.60f;
    float ratchet_lock_phase_deadband_m = 0.005f;
    float ratchet_lock_phase_initial_front_error_m = 0.0f;
    float ratchet_lock_phase_max_backoff_m = 0.030f;
    float ratchet_lock_phase_max_forward_m = 0.015f;
    float ratchet_lock_phase_max_error_m = 0.080f;
    int ratchet_soft_upper_ttl_steps = 5;
    bool ratchet_first_collision_enters_stair_mode = true;
    bool ratchet_single_collision_confirms_interval = false;
    bool ratchet_two_collision_enabled = true;
    float ratchet_two_collision_interval_margin_m = 0.025f;
    float ratchet_two_collision_stride_layers = 2.0f;
    int ratchet_two_collision_min_layer_delta = 1;
    float ratchet_two_collision_min_height_delta_m = 0.055f;
    float ratchet_two_collision_nominal_riser_height_m = 0.15f;
    float ratchet_two_collision_height_layer_tolerance_m = 0.06f;
    bool ratchet_two_collision_use_height_layers = true;
    float ratchet_two_collision_lower_cross_margin_m = 0.10f;
    float ratchet_two_collision_upper_cross_margin_m = 0.08f;
    int ratchet_rejected_second_hit_confirm_count = 2;
    float ratchet_rejected_second_hit_target_tolerance_m = 0.05f;
    bool ratchet_second_collision_requires_up_step = true;
    float ratchet_two_collision_tread_min_m = 0.23f;
    float ratchet_two_collision_tread_max_m = 0.37f;
    bool ratchet_anchor_collision_enabled = true;
    float ratchet_lower_target_lag_margin_m = 0.0f;
    float ratchet_same_foot_stride_guard_layers = 2.0f;
    float ratchet_same_foot_stride_guard_margin_m = 0.04f;
    float ratchet_collision_min_confidence = 0.45f;
    float ratchet_post_first_collision_collision_min_confidence = 0.30f;
    float ratchet_post_first_collision_probe_increment_m = 0.05f;
    float ratchet_min_interval_width_m = 0.04f;
    float ratchet_min_stride_m = 0.10f;
    float ratchet_max_stride_m = 0.85f;
    int ratchet_reset_flat_pairs = 4;

    float odom_contact_lock_threshold = 0.55f;
    float odom_contact_release_threshold = 0.20f;
    float odom_max_step_translation_m = 0.20f;
    float odom_max_double_support_residual_m = 0.06f;
};

struct FootEventMemoryFrame
{
    int step = 0;
    float dt = 0.02f;
    float phase = 0.0f;
    std::array<float, 3> command = {0.0f, 0.0f, 0.0f};
    Eigen::Quaternionf root_quat_w = Eigen::Quaternionf::Identity();
    Eigen::Vector3f left_toe_pos_b = Eigen::Vector3f::Zero();
    Eigen::Vector3f right_toe_pos_b = Eigen::Vector3f::Zero();
    Eigen::Vector3f left_heel_pos_b = Eigen::Vector3f::Zero();
    Eigen::Vector3f right_heel_pos_b = Eigen::Vector3f::Zero();
    std::array<float, 2> contact_prob = {0.0f, 0.0f};
    std::array<float, 2> touchdown_prob = {0.0f, 0.0f};
    std::array<float, 2> toe_riser_prob = {0.0f, 0.0f};
    std::array<bool, 2> touchdown_event = {false, false};
    std::array<bool, 2> toe_riser_event = {false, false};
};

class FootEventMemory
{
public:
    static constexpr int kFootprintSlotDim = 17;
    static constexpr int kToeMarkSlotDim = 16;
    static constexpr int kPairCount = 5;
    static constexpr int kPairFeatureDim = 10;
    static constexpr int kToeSummaryStart = 50;
    static constexpr int kGeometryStatsStart = 60;
    static constexpr int kRatchetStart = 70;
    static constexpr int kSummaryDim = 80;

    enum class Mode
    {
        Probe = 0,
        Backoff = 1,
        Lock = 2,
    };

    void configure(const FootEventMemoryConfig& cfg)
    {
        cfg_ = cfg;
        cfg_.memory_len = std::max(1, cfg_.memory_len);
        cfg_.age_norm_s = std::max(cfg_.age_norm_s, 1.0e-6f);
        cfg_.stance_age_norm_s = std::max(cfg_.stance_age_norm_s, 1.0e-6f);
        cfg_.predicted_fill_grace_frames =
            std::max(1, cfg_.predicted_fill_grace_frames);
        cfg_.predicted_fill_phase_window =
            clamp(cfg_.predicted_fill_phase_window, 0.0f, 0.5f);
        cfg_.predicted_fill_phase_lead_window =
            clamp(cfg_.predicted_fill_phase_lead_window, 0.0f, 0.5f);
        cfg_.predicted_fill_contact_threshold =
            clamp(cfg_.predicted_fill_contact_threshold, 0.0f, 1.0f);
        cfg_.predicted_fill_min_command_norm =
            std::max(0.0f, cfg_.predicted_fill_min_command_norm);
        cfg_.predicted_fill_confidence =
            clamp(cfg_.predicted_fill_confidence, 0.0f, 1.0f);
        cfg_.predicted_fill_touchdown_prob =
            clamp(cfg_.predicted_fill_touchdown_prob, 0.0f, 1.0f);
        cfg_.touchdown_gate_contact_threshold =
            clamp(cfg_.touchdown_gate_contact_threshold, 0.0f, 1.0f);
        cfg_.ratchet_probe_cap_m = clamp(
            cfg_.ratchet_probe_cap_m,
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);
        reset();
    }

    void reset()
    {
        footprints_.assign(
            static_cast<size_t>(cfg_.memory_len),
            std::array<float, kFootprintSlotDim>{});
        toe_marks_.assign(
            static_cast<size_t>(cfg_.memory_len),
            std::array<float, kToeMarkSlotDim>{});
        footprint_valid_.assign(static_cast<size_t>(cfg_.memory_len), false);
        toe_valid_.assign(static_cast<size_t>(cfg_.memory_len), false);
        footprint_pos_w_.assign(
            static_cast<size_t>(cfg_.memory_len),
            Eigen::Vector3f::Zero());
        toe_pos_w_.assign(
            static_cast<size_t>(cfg_.memory_len),
            Eigen::Vector3f::Zero());
        foot_in_stance_ = {false, false};
        release_count_ = {0, 0};
        stance_age_s_ = {0.0f, 0.0f};
        active_footprint_slot_ = {-1, -1};
        predicted_fill_state_ = {PredictedFillState{}, PredictedFillState{}};
        odom_ = ContactFootOdometry{};
        ratchet_ = RatchetState{};
        summary_.assign(kSummaryDim, 0.0f);
    }

    const std::vector<float>& update(const FootEventMemoryFrame& frame)
    {
        const std::array<Eigen::Vector3f, 2> foot_points = {{
            0.5f * (frame.left_toe_pos_b + frame.left_heel_pos_b),
            0.5f * (frame.right_toe_pos_b + frame.right_heel_pos_b),
        }};
        const std::array<Eigen::Vector3f, 2> toe_points = {{
            frame.left_toe_pos_b,
            frame.right_toe_pos_b,
        }};

        update_stance_latches(frame.contact_prob, frame.dt);

        const std::array<float, 2> phase_now = phase_features(frame);
        bool new_footprint_any = false;
        bool new_toe_mark_any = false;

        refresh_predicted_fill_phase(frame.phase);

        std::array<bool, 2> accepted_touchdown = {false, false};
        std::array<bool, 2> predicted_touchdown = {false, false};
        for (int foot = 0; foot < 2; ++foot) {
            const bool allowed = !foot_in_stance_[foot];
            accepted_touchdown[foot] =
                frame.touchdown_event[foot] &&
                allowed &&
                touchdown_event_allowed(frame, foot);
            if (!accepted_touchdown[foot]) {
                predicted_touchdown[foot] =
                    should_insert_predicted_footprint(frame, foot);
            }
        }

        std::array<bool, 2> odom_touchdown = {{
            accepted_touchdown[0] || (predicted_touchdown[0] && !odom_.locked[0]),
            accepted_touchdown[1] || (predicted_touchdown[1] && !odom_.locked[1]),
        }};
        odom_.update(
            cfg_,
            frame.root_quat_w,
            foot_points,
            frame.contact_prob,
            odom_touchdown);
        age_and_refresh(frame.dt, frame.root_quat_w);

        for (int foot = 0; foot < 2; ++foot) {
            if (accepted_touchdown[foot]) {
                push_footprint(
                    foot,
                    foot_points[foot],
                    frame.root_quat_w,
                    phase_now,
                    frame.contact_prob[foot],
                    frame.touchdown_prob[foot],
                    frame.touchdown_prob[foot],
                    false);
                predicted_fill_state_[foot].pending = false;
                new_footprint_any = true;
            } else if (predicted_touchdown[foot]) {
                push_footprint(
                    foot,
                    foot_points[foot],
                    frame.root_quat_w,
                    phase_now,
                    frame.contact_prob[foot],
                    cfg_.predicted_fill_touchdown_prob,
                    cfg_.predicted_fill_confidence,
                    true);
                predicted_fill_state_[foot].pending = false;
                new_footprint_any = true;
            }

            const bool toe = frame.toe_riser_event[foot];
            if (toe) {
                const bool swing_or_early_contact =
                    frame.contact_prob[foot] < cfg_.odom_contact_lock_threshold ||
                    stance_age_s_[foot] <= cfg_.early_contact_time_s;
                push_toe_mark(
                    foot,
                    toe_points[foot],
                    frame.root_quat_w,
                    phase_now,
                    frame.contact_prob[foot],
                    frame.toe_riser_prob[foot],
                    frame.toe_riser_prob[foot],
                    swing_or_early_contact);
                new_toe_mark_any = true;
            }
        }

        for (int foot = 0; foot < 2; ++foot) {
            if (frame.contact_prob[foot] >= cfg_.odom_contact_lock_threshold &&
                !foot_in_stance_[foot] &&
                !predicted_fill_state_[foot].pending &&
                !predicted_fill_phase_guard(frame.phase, foot) &&
                !frame.touchdown_event[foot]) {
                foot_in_stance_[foot] = true;
                stance_age_s_[foot] = 0.0f;
            }
        }

        refresh_relative_positions(frame.root_quat_w);
        compute_summary(true, new_footprint_any, new_toe_mark_any, frame.dt);
        return summary_;
    }

    const std::vector<float>& summary() const
    {
        return summary_;
    }

#ifdef MJLAB_FOOT_EVENT_MEMORY_TESTING
    Eigen::Vector3f odom_base_pos_for_testing() const
    {
        return odom_.base_pos_w;
    }

    bool odom_locked_for_testing(int foot) const
    {
        return foot >= 0 && foot < 2 && odom_.locked[foot];
    }

    std::array<float, kFootprintSlotDim> footprint_for_testing(int slot) const
    {
        if (slot < 0 || slot >= static_cast<int>(footprints_.size())) {
            return std::array<float, kFootprintSlotDim>{};
        }
        return footprints_[static_cast<size_t>(slot)];
    }

    size_t valid_footprint_count_for_testing() const
    {
        return static_cast<size_t>(std::count(
            footprint_valid_.begin(),
            footprint_valid_.end(),
            true));
    }

    bool predicted_fill_pending_for_testing(int foot) const
    {
        return foot >= 0 && foot < 2 && predicted_fill_state_[foot].pending;
    }
#endif

private:
    struct ContactFootOdometry
    {
        Eigen::Vector3f base_pos_w = Eigen::Vector3f::Zero();
        std::array<bool, 2> locked = {false, false};
        std::array<int, 2> release_count = {0, 0};
        std::array<Eigen::Vector3f, 2> locked_foot_w = {{
            Eigen::Vector3f::Zero(),
            Eigen::Vector3f::Zero(),
        }};

        void update(
            const FootEventMemoryConfig& cfg,
            const Eigen::Quaternionf& root_quat_w,
            const std::array<Eigen::Vector3f, 2>& foot_points_b,
            const std::array<float, 2>& contact_prob,
            const std::array<bool, 2>& touchdown_event)
        {
            for (int foot = 0; foot < 2; ++foot) {
                const bool lock_now =
                    touchdown_event[foot] ||
                    (!locked[foot] && contact_prob[foot] >= cfg.odom_contact_lock_threshold);
                if (lock_now) {
                    locked[foot] = true;
                    release_count[foot] = 0;
                    locked_foot_w[foot] = base_pos_w + root_quat_w * foot_points_b[foot];
                }
                if (locked[foot] && contact_prob[foot] < cfg.odom_contact_release_threshold) {
                    release_count[foot] += 1;
                    if (release_count[foot] >= cfg.release_confirm_frames) {
                        locked[foot] = false;
                    }
                } else if (locked[foot]) {
                    release_count[foot] = 0;
                }
            }

            std::array<Eigen::Vector3f, 2> estimates = {{
                Eigen::Vector3f::Zero(),
                Eigen::Vector3f::Zero(),
            }};
            Eigen::Vector3f estimate = Eigen::Vector3f::Zero();
            int count = 0;
            for (int foot = 0; foot < 2; ++foot) {
                if (!locked[foot] || contact_prob[foot] < cfg.odom_contact_release_threshold) {
                    continue;
                }
                estimates[foot] = locked_foot_w[foot] - root_quat_w * foot_points_b[foot];
                estimate += estimates[foot];
                count += 1;
            }
            if (count <= 0) {
                return;
            }
            if (count == 2) {
                const float residual = (estimates[0] - estimates[1]).norm();
                if (residual > cfg.odom_max_double_support_residual_m) {
                    const float left_jump = (estimates[0] - base_pos_w).norm();
                    const float right_jump = (estimates[1] - base_pos_w).norm();
                    if (std::abs(left_jump - right_jump) <=
                        0.5f * cfg.odom_max_double_support_residual_m) {
                        return;
                    }
                    estimate = left_jump < right_jump ? estimates[0] : estimates[1];
                } else {
                    estimate *= 0.5f;
                }
            } else {
                estimate /= static_cast<float>(count);
            }
            const float jump = (estimate - base_pos_w).norm();
            if (jump <= cfg.odom_max_step_translation_m) {
                base_pos_w = estimate;
            }
        }
    };

    struct PredictedFillState
    {
        bool prev_phase_guard = false;
        bool pending = false;
        int age_frames = 0;
        int expected_age_frames = 0;
    };

    struct RatchetState
    {
        bool active = false;
        float lower_s = 0.0f;
        float probe_target_s = 0.0f;
        float upper_s = 0.0f;
        float collision_upper_s = 0.0f;
        float same_foot_stride_lower_s = 0.0f;
        float same_foot_stride_upper_s = 0.0f;
        float validated_stride_s = 0.0f;
        float validated_riser_height_s = 0.0f;
        bool probe_bootstrap_ready = false;
        bool probe_bootstrap_provisional = false;
        bool interval_confirmed = false;
        bool first_collision_seen = false;
        bool collision_anchor_active = false;
        float collision_anchor_s = 0.0f;
        float collision_anchor_z = 0.0f;
        int collision_anchor_up_steps = 0;
        float last_forward_up_stride = 0.0f;
        float stride_growth = 0.0f;
        float last_forward_up_height = 0.0f;
        int safe_no_hit_steps = 0;
        float confidence = 0.0f;
        float age_s = 0.0f;
        int flat_pair_steps = 0;
        bool soft_upper_active = false;
        int soft_upper_ttl_remaining = 0;
        bool upper_seen = false;
        Mode mode = Mode::Probe;
        bool recovery_pending = false;
        bool recovery_attempt_seen = false;
        bool recovery_reduction_seen = false;
        int recovery_pending_steps = 0;
        float recovery_target_s = 0.0f;
        float lock_target_s = 0.0f;
        int lock_stable_count = 0;
        float lock_lower_s = 0.0f;
        float lock_upper_s = 0.0f;
        float lock_phase_error_s = 0.0f;
        float lock_phase_correction_s = 0.0f;
        int rejected_second_hit_count = 0;
        float rejected_second_hit_target_s = 0.0f;
        bool hold_center_active = false;
        float hold_center_target_s = 0.0f;
    };

    using Footprint = std::array<float, kFootprintSlotDim>;
    using ToeMark = std::array<float, kToeMarkSlotDim>;
    using PairFeatures = std::array<float, kPairFeatureDim>;
    using ToeFeatures = std::array<float, 10>;
    using SameFootFeatures = std::array<float, 5>;

    static float clamp(float value, float low, float high)
    {
        return std::max(low, std::min(value, high));
    }

    static float finite_or_zero(float value)
    {
        return std::isfinite(value) ? value : 0.0f;
    }

    static Eigen::Quaternionf yaw_quat(const Eigen::Quaternionf& quat)
    {
        const float siny_cosp = 2.0f * (quat.w() * quat.z() + quat.x() * quat.y());
        const float cosy_cosp =
            1.0f - 2.0f * (quat.y() * quat.y() + quat.z() * quat.z());
        const float yaw = std::atan2(siny_cosp, cosy_cosp);
        return Eigen::Quaternionf(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
    }

    Eigen::Vector3f world_to_current_base_yaw(
        const Eigen::Vector3f& point_w,
        const Eigen::Quaternionf& root_quat_w) const
    {
        return yaw_quat(root_quat_w).conjugate() * (point_w - odom_.base_pos_w);
    }

    std::array<float, 2> phase_features(const FootEventMemoryFrame& frame) const
    {
        const float norm = std::sqrt(
            frame.command[0] * frame.command[0] +
            frame.command[1] * frame.command[1] +
            frame.command[2] * frame.command[2]);
        if (norm < 0.1f) {
            return {0.0f, 0.0f};
        }
        const float angle = frame.phase * 2.0f * static_cast<float>(M_PI);
        return {std::sin(angle), std::cos(angle)};
    }

    static float wrap_phase(float phase)
    {
        float wrapped = std::fmod(phase, 1.0f);
        if (wrapped < 0.0f) {
            wrapped += 1.0f;
        }
        return wrapped;
    }

    float foot_stance_phase(float phase, int foot) const
    {
        return wrap_phase(phase + (foot == 1 ? 0.5f : 0.0f));
    }

    bool expected_stance_phase(float phase, int foot) const
    {
        if (cfg_.predicted_fill_phase_window <= 0.0f) {
            return false;
        }
        return foot_stance_phase(phase, foot) < cfg_.predicted_fill_phase_window;
    }

    bool pre_stance_phase(float phase, int foot) const
    {
        if (cfg_.predicted_fill_phase_lead_window <= 0.0f) {
            return false;
        }
        return foot_stance_phase(phase, foot) >
               1.0f - cfg_.predicted_fill_phase_lead_window;
    }

    bool touchdown_phase_allowed(float phase, int foot) const
    {
        return expected_stance_phase(phase, foot) || pre_stance_phase(phase, foot);
    }

    bool predicted_fill_phase_guard(float phase, int foot) const
    {
        return cfg_.predicted_fill_enabled && touchdown_phase_allowed(phase, foot);
    }

    void refresh_predicted_fill_phase(float phase)
    {
        for (int foot = 0; foot < 2; ++foot) {
            auto& state = predicted_fill_state_[foot];
            const bool guard = predicted_fill_phase_guard(phase, foot);
            const bool expected = expected_stance_phase(phase, foot);
            if (!cfg_.predicted_fill_enabled || !guard) {
                state.pending = false;
                state.age_frames = 0;
                state.expected_age_frames = 0;
                state.prev_phase_guard = guard;
                continue;
            }
            if (guard && !state.prev_phase_guard && !foot_in_stance_[foot]) {
                state.pending = true;
                state.age_frames = 0;
                state.expected_age_frames = 0;
            }
            if (state.pending) {
                state.age_frames += 1;
                if (expected) {
                    state.expected_age_frames += 1;
                }
            }
            state.prev_phase_guard = guard;
        }
    }

    bool touchdown_event_allowed(const FootEventMemoryFrame& frame, int foot) const
    {
        if (!cfg_.touchdown_gate_enabled) {
            return true;
        }
        if (frame.contact_prob[foot] < cfg_.touchdown_gate_contact_threshold) {
            return false;
        }
        return touchdown_phase_allowed(frame.phase, foot);
    }

    bool should_insert_predicted_footprint(const FootEventMemoryFrame& frame, int foot)
    {
        if (!cfg_.predicted_fill_enabled || foot_in_stance_[foot]) {
            return false;
        }
        auto& state = predicted_fill_state_[foot];
        if (!state.pending) {
            return false;
        }
        if (!expected_stance_phase(frame.phase, foot) ||
            state.expected_age_frames < cfg_.predicted_fill_grace_frames) {
            return false;
        }
        const float command_norm = std::sqrt(
            frame.command[0] * frame.command[0] +
            frame.command[1] * frame.command[1] +
            frame.command[2] * frame.command[2]);
        if (command_norm < cfg_.predicted_fill_min_command_norm) {
            return false;
        }
        return frame.contact_prob[foot] >= cfg_.predicted_fill_contact_threshold;
    }

    void age_and_refresh(float dt, const Eigen::Quaternionf& root_quat_w)
    {
        const float age_delta = dt / cfg_.age_norm_s;
        for (int i = 0; i < cfg_.memory_len; ++i) {
            if (footprint_valid_[i]) {
                footprints_[i][6] = clamp(footprints_[i][6] + age_delta, 0.0f, 1.0f);
                if (footprints_[i][6] >= 1.0f) {
                    footprint_valid_[i] = false;
                    footprints_[i] = Footprint{};
                    footprint_pos_w_[i] = Eigen::Vector3f::Zero();
                }
            }
            if (toe_valid_[i]) {
                toe_marks_[i][5] = clamp(toe_marks_[i][5] + age_delta, 0.0f, 1.0f);
                if (toe_marks_[i][5] >= 1.0f) {
                    toe_valid_[i] = false;
                    toe_marks_[i] = ToeMark{};
                    toe_pos_w_[i] = Eigen::Vector3f::Zero();
                }
            }
        }
        for (int foot = 0; foot < 2; ++foot) {
            const int slot = active_footprint_slot_[foot];
            if (slot < 0 || slot >= cfg_.memory_len || !footprint_valid_[slot]) {
                active_footprint_slot_[foot] = -1;
            }
        }
        refresh_relative_positions(root_quat_w);
    }

    void refresh_relative_positions(const Eigen::Quaternionf& root_quat_w)
    {
        for (int i = 0; i < cfg_.memory_len; ++i) {
            if (footprint_valid_[i]) {
                const Eigen::Vector3f rel =
                    world_to_current_base_yaw(footprint_pos_w_[i], root_quat_w);
                footprints_[i][7] = rel.x();
                footprints_[i][8] = rel.y();
                footprints_[i][9] = rel.z();
                footprints_[i][10] = rel.x();
                footprints_[i][11] = rel.y();
            } else {
                for (int j = 7; j <= 11; ++j) {
                    footprints_[i][j] = 0.0f;
                }
            }

            if (toe_valid_[i]) {
                const Eigen::Vector3f rel =
                    world_to_current_base_yaw(toe_pos_w_[i], root_quat_w);
                toe_marks_[i][6] = rel.x();
                toe_marks_[i][7] = rel.y();
                toe_marks_[i][8] = rel.z();
                toe_marks_[i][9] = rel.x();
                toe_marks_[i][10] = rel.y();
            } else {
                for (int j = 6; j <= 10; ++j) {
                    toe_marks_[i][j] = 0.0f;
                }
            }
        }
    }

    void update_stance_latches(const std::array<float, 2>& contact_prob, float dt)
    {
        for (int foot = 0; foot < 2; ++foot) {
            if (contact_prob[foot] < cfg_.release_contact_prob_threshold) {
                release_count_[foot] += 1;
            } else {
                release_count_[foot] = 0;
            }
            if (foot_in_stance_[foot] &&
                release_count_[foot] >= cfg_.release_confirm_frames) {
                foot_in_stance_[foot] = false;
                active_footprint_slot_[foot] = -1;
            }
            if (foot_in_stance_[foot]) {
                stance_age_s_[foot] += dt;
            }
            const int slot = active_footprint_slot_[foot];
            if (slot >= 0 && slot < cfg_.memory_len && footprint_valid_[slot]) {
                footprints_[slot][16] =
                    clamp(stance_age_s_[foot] / cfg_.stance_age_norm_s, 0.0f, 1.0f);
            }
        }
    }

    void shift_footprints()
    {
        for (int i = cfg_.memory_len - 1; i >= 1; --i) {
            footprints_[i] = footprints_[i - 1];
            footprint_valid_[i] = footprint_valid_[i - 1];
            footprint_pos_w_[i] = footprint_pos_w_[i - 1];
        }
        for (int foot = 0; foot < 2; ++foot) {
            if (active_footprint_slot_[foot] >= 0) {
                active_footprint_slot_[foot] += 1;
                if (active_footprint_slot_[foot] >= cfg_.memory_len) {
                    active_footprint_slot_[foot] = -1;
                }
            }
        }
    }

    void shift_toe_marks()
    {
        for (int i = cfg_.memory_len - 1; i >= 1; --i) {
            toe_marks_[i] = toe_marks_[i - 1];
            toe_valid_[i] = toe_valid_[i - 1];
            toe_pos_w_[i] = toe_pos_w_[i - 1];
        }
    }

    void push_footprint(
        int foot,
        const Eigen::Vector3f& point_body,
        const Eigen::Quaternionf& root_quat_w,
        const std::array<float, 2>& phase_now,
        float contact_prob,
        float touchdown_prob,
        float confidence,
        bool source_predicted_fill = false)
    {
        shift_footprints();
        const Eigen::Vector3f point_w = odom_.base_pos_w + root_quat_w * point_body;
        const Eigen::Vector3f rel = world_to_current_base_yaw(point_w, root_quat_w);

        Footprint feature{};
        feature[0] = 1.0f;
        feature[1] = foot == 0 ? 1.0f : 0.0f;
        feature[2] = foot == 1 ? 1.0f : 0.0f;
        feature[3] = source_predicted_fill ? 0.0f : 1.0f;
        feature[4] = source_predicted_fill ? 1.0f : 0.0f;
        feature[5] = clamp(confidence, 0.0f, 1.0f);
        feature[6] = 0.0f;
        feature[7] = rel.x();
        feature[8] = rel.y();
        feature[9] = rel.z();
        feature[10] = rel.x();
        feature[11] = rel.y();
        feature[12] = phase_now[0];
        feature[13] = phase_now[1];
        feature[14] = clamp(contact_prob, 0.0f, 1.0f);
        feature[15] = clamp(touchdown_prob, 0.0f, 1.0f);
        feature[16] = 0.0f;
        footprints_[0] = feature;
        footprint_valid_[0] = true;
        footprint_pos_w_[0] = point_w;
        foot_in_stance_[foot] = true;
        release_count_[foot] = 0;
        stance_age_s_[foot] = 0.0f;
        active_footprint_slot_[foot] = 0;
    }

    void push_toe_mark(
        int foot,
        const Eigen::Vector3f& point_body,
        const Eigen::Quaternionf& root_quat_w,
        const std::array<float, 2>& phase_now,
        float contact_prob,
        float toe_prob,
        float confidence,
        bool swing_or_early_contact)
    {
        shift_toe_marks();
        const Eigen::Vector3f point_w = odom_.base_pos_w + root_quat_w * point_body;
        const Eigen::Vector3f rel = world_to_current_base_yaw(point_w, root_quat_w);

        ToeMark feature{};
        feature[0] = 1.0f;
        feature[1] = foot == 0 ? 1.0f : 0.0f;
        feature[2] = foot == 1 ? 1.0f : 0.0f;
        feature[3] = 1.0f;
        feature[4] = clamp(confidence, 0.0f, 1.0f);
        feature[5] = 0.0f;
        feature[6] = rel.x();
        feature[7] = rel.y();
        feature[8] = rel.z();
        feature[9] = rel.x();
        feature[10] = rel.y();
        feature[11] = phase_now[0];
        feature[12] = phase_now[1];
        feature[13] = clamp(toe_prob, 0.0f, 1.0f);
        feature[14] = clamp(contact_prob, 0.0f, 1.0f);
        feature[15] = swing_or_early_contact ? 1.0f : 0.0f;
        toe_marks_[0] = feature;
        toe_valid_[0] = true;
        toe_pos_w_[0] = point_w;
    }

    PairFeatures pair_features(const Footprint& newer, const Footprint& older, bool valid) const
    {
        PairFeatures f{};
        const bool newer_left = newer[1] > 0.5f;
        const bool newer_right = newer[2] > 0.5f;
        const bool older_left = older[1] > 0.5f;
        const bool alternating = newer_left != older_left;
        const bool both_confirmed = newer[3] > 0.5f && older[3] > 0.5f;
        const float min_conf = clamp(std::min(newer[5], older[5]), 0.0f, 1.0f);
        const float age_max = clamp(std::max(newer[6], older[6]), 0.0f, 1.0f);
        const float delta_s = newer[10] - older[10];
        const float delta_z = newer[9] - older[9];
        const float lateral_delta = newer[11] - older[11];
        const float source_score = both_confirmed ? 1.0f : 0.55f;
        const float score = valid ? min_conf * (1.0f - age_max) * source_score : 0.0f;
        const float foot_sign = newer_right ? 1.0f : (newer_left ? -1.0f : 0.0f);
        f[0] = valid ? 1.0f : 0.0f;
        f[1] = score;
        f[2] = valid ? foot_sign : 0.0f;
        f[3] = valid && alternating ? 1.0f : 0.0f;
        f[4] = age_max;
        f[5] = valid ? delta_s : 0.0f;
        f[6] = valid ? delta_z : 0.0f;
        f[7] = valid ? lateral_delta : 0.0f;
        f[8] = valid ? std::abs(delta_s) : 0.0f;
        f[9] = valid ? min_conf : 0.0f;
        return f;
    }

    std::array<PairFeatures, kPairCount> all_pair_features() const
    {
        std::array<PairFeatures, kPairCount> pairs{};
        const int count = std::min(kPairCount, std::max(0, cfg_.memory_len - 1));
        for (int i = 0; i < count; ++i) {
            pairs[i] = pair_features(
                footprints_[i],
                footprints_[i + 1],
                footprint_valid_[i] && footprint_valid_[i + 1]);
        }
        return pairs;
    }

    SameFootFeatures same_foot_step_feature(int newer_idx) const
    {
        SameFootFeatures out{};
        if (newer_idx < 0 || newer_idx >= cfg_.memory_len || !footprint_valid_[newer_idx]) {
            return out;
        }
        const auto& newer = footprints_[newer_idx];
        const bool newer_left = newer[1] > 0.5f;
        const bool newer_right = newer[2] > 0.5f;
        int selected = -1;
        for (int i = newer_idx + 1; i < cfg_.memory_len; ++i) {
            if (!footprint_valid_[i]) {
                continue;
            }
            const bool older_left = footprints_[i][1] > 0.5f;
            const bool older_right = footprints_[i][2] > 0.5f;
            if ((newer_left && older_left) || (newer_right && older_right)) {
                selected = i;
                break;
            }
        }
        if (selected < 0) {
            return out;
        }
        const auto& older = footprints_[selected];
        const bool both_confirmed = newer[3] > 0.5f && older[3] > 0.5f;
        const float min_conf = clamp(std::min(newer[5], older[5]), 0.0f, 1.0f);
        const float age_max = clamp(std::max(newer[6], older[6]), 0.0f, 1.0f);
        const float source_score = both_confirmed ? 1.0f : 0.55f;
        out[0] = 1.0f;
        out[1] = min_conf * (1.0f - age_max) * source_score;
        out[2] = newer[10] - older[10];
        out[3] = newer[9] - older[9];
        out[4] = age_max;
        return out;
    }

    ToeFeatures latest_toe_features() const
    {
        ToeFeatures out{};
        if (cfg_.memory_len <= 0 || !toe_valid_[0]) {
            return out;
        }
        const auto& toe = toe_marks_[0];
        const bool toe_left = toe[1] > 0.5f;
        const bool toe_right = toe[2] > 0.5f;
        const float toe_s = toe[9];
        const float toe_age = toe[5];
        int same_before = -1;
        int opposite_after = -1;
        float same_before_age = 2.0f;
        float opposite_after_age = -1.0f;
        for (int i = 0; i < cfg_.memory_len; ++i) {
            if (!footprint_valid_[i]) {
                continue;
            }
            const bool fp_left = footprints_[i][1] > 0.5f;
            const bool fp_right = footprints_[i][2] > 0.5f;
            const bool same = (toe_left && fp_left) || (toe_right && fp_right);
            const bool opposite = (toe_left && fp_right) || (toe_right && fp_left);
            if (same && footprints_[i][6] > toe_age + 1.0e-6f &&
                footprints_[i][6] < same_before_age) {
                same_before = i;
                same_before_age = footprints_[i][6];
            }
            if (opposite && footprints_[i][6] < toe_age - 1.0e-6f &&
                footprints_[i][6] > opposite_after_age) {
                opposite_after = i;
                opposite_after_age = footprints_[i][6];
            }
        }
        out[0] = 1.0f;
        out[1] = clamp(toe[4], 0.0f, 1.0f);
        out[2] = toe[5];
        out[3] = toe_right ? 1.0f : (toe_left ? -1.0f : 0.0f);
        out[4] = toe_s;
        out[5] = toe[8];
        if (same_before >= 0) {
            out[6] = 1.0f;
            out[7] = toe_s - footprints_[same_before][10];
        }
        if (opposite_after >= 0) {
            out[8] = 1.0f;
            out[9] = footprints_[opposite_after][10] - toe_s;
        }
        return out;
    }

    std::array<float, 10> summary_stats(
        const std::array<PairFeatures, kPairCount>& pairs) const
    {
        std::array<float, 10> stats{};
        float count = 0.0f;
        float weight_sum = 0.0f;
        float weighted_s = 0.0f;
        float weighted_z = 0.0f;
        bool forward_any = false;
        bool forward_up_any = false;
        float max_forward = 0.0f;
        float max_forward_up = 0.0f;
        float positive_weight_sum = 0.0f;
        float positive_height_sum = 0.0f;
        for (const auto& p : pairs) {
            const bool valid = p[0] > 0.5f;
            if (!valid) {
                continue;
            }
            count += 1.0f;
            const float weight = p[1];
            weight_sum += weight;
            weighted_s += p[5] * weight;
            weighted_z += p[6] * weight;
            if (p[5] > 0.0f) {
                forward_any = true;
                max_forward = std::max(max_forward, p[5]);
            }
            if (p[5] > 0.0f && p[6] > cfg_.ratchet_height_threshold_m) {
                forward_up_any = true;
                max_forward_up = std::max(max_forward_up, p[5]);
                positive_weight_sum += weight;
                positive_height_sum += p[6] * weight;
            }
        }
        stats[0] = clamp(count / static_cast<float>(kPairCount), 0.0f, 1.0f);
        if (count > 0.0f && weight_sum > 1.0e-6f) {
            stats[1] = weighted_s / weight_sum;
            stats[2] = weighted_z / weight_sum;
        }
        if (pairs[0][0] > 0.5f) {
            stats[3] = pairs[0][5];
            stats[4] = pairs[0][6];
        }
        if (pairs[0][0] > 0.5f && pairs[1][0] > 0.5f) {
            stats[5] = pairs[0][5] - pairs[1][5];
            stats[6] = pairs[0][6] - pairs[1][6];
        }
        stats[7] = forward_any ? max_forward : 0.0f;
        stats[8] = forward_up_any ? max_forward_up : 0.0f;
        stats[9] = positive_weight_sum > 1.0e-6f
            ? positive_height_sum / positive_weight_sum
            : 0.0f;
        return stats;
    }

    struct SequenceFeatures
    {
        bool valid = false;
        float score = 0.0f;
        float same_foot_stride = 0.0f;
        float same_foot_rise = 0.0f;
        float first_rise = 0.0f;
        float second_rise = 0.0f;
    };

    SequenceFeatures latest_sequence_features() const
    {
        SequenceFeatures seq;
        if (cfg_.memory_len < 3 ||
            !footprint_valid_[0] || !footprint_valid_[1] || !footprint_valid_[2]) {
            return seq;
        }
        const auto& newer = footprints_[0];
        const auto& middle = footprints_[1];
        const auto& older = footprints_[2];
        const bool newer_left = newer[1] > 0.5f;
        const bool middle_left = middle[1] > 0.5f;
        const bool older_left = older[1] > 0.5f;
        const bool newer_right = newer[2] > 0.5f;
        const bool older_right = older[2] > 0.5f;
        const bool alternating = (newer_left != middle_left) && (middle_left != older_left);
        const bool same_outer = (newer_left && older_left) || (newer_right && older_right);
        const float first_stride = middle[10] - older[10];
        const float second_stride = newer[10] - middle[10];
        seq.same_foot_stride = newer[10] - older[10];
        seq.first_rise = middle[9] - older[9];
        seq.second_rise = newer[9] - middle[9];
        seq.same_foot_rise = newer[9] - older[9];
        const float confidence = clamp(std::min(newer[5], std::min(middle[5], older[5])), 0.0f, 1.0f);
        const bool ordered_forward =
            first_stride >= cfg_.ratchet_min_stride_m &&
            second_stride >= cfg_.ratchet_min_stride_m;
        const bool ordered_up =
            seq.first_rise > cfg_.ratchet_height_threshold_m &&
            seq.second_rise > cfg_.ratchet_height_threshold_m;
        const bool bounded =
            seq.first_rise <= cfg_.ratchet_sequence_max_adjacent_height_m &&
            seq.second_rise <= cfg_.ratchet_sequence_max_adjacent_height_m;
        const bool consistent =
            std::abs(seq.first_rise - seq.second_rise) <=
            cfg_.ratchet_sequence_height_tolerance_m;
        bool reference_ok = consistent;
        if (ratchet_.validated_riser_height_s > cfg_.ratchet_height_threshold_m) {
            const float tol = std::min(
                cfg_.ratchet_sequence_height_tolerance_m,
                std::max(0.025f, 0.5f * ratchet_.validated_riser_height_s));
            reference_ok =
                std::abs(seq.first_rise - ratchet_.validated_riser_height_s) <= tol &&
                std::abs(seq.second_rise - ratchet_.validated_riser_height_s) <= tol;
        }
        seq.valid =
            alternating && same_outer && ordered_forward && ordered_up &&
            bounded && reference_ok && confidence >= cfg_.ratchet_sequence_min_confidence;
        seq.score = seq.valid ? confidence : 0.0f;
        return seq;
    }

    struct IntervalEstimate
    {
        bool valid = false;
        float lower = 0.0f;
        float center = 0.0f;
        float upper = 0.0f;
        float tread_depth = 0.0f;
        float layer_delta = 0.0f;
    };

    struct AnchorGeometry
    {
        bool collision_valid = false;
        bool second_hit_gate = false;
        bool second_riser_gate = false;
        bool fallback = false;
        IntervalEstimate collision;
        IntervalEstimate fallback_interval;
    };

    IntervalEstimate make_interval(float center, float phase_stride_guard_cap, float min_width) const
    {
        IntervalEstimate est;
        est.center = clamp(center, cfg_.ratchet_min_stride_m, cfg_.ratchet_max_stride_m);
        est.upper = clamp(
            std::min(
                est.center + cfg_.ratchet_two_collision_interval_margin_m,
                phase_stride_guard_cap),
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);
        const float lower_raw = clamp(
            est.center - cfg_.ratchet_two_collision_interval_margin_m,
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);
        est.lower = clamp(
            std::min(lower_raw, est.upper - min_width),
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);
        est.center = 0.5f * (est.lower + est.upper);
        est.valid = est.upper >= est.lower + min_width;
        return est;
    }

    static bool target_contains(
        float value,
        float target,
        float lower_tolerance,
        float upper_tolerance)
    {
        return value >= target - lower_tolerance &&
               value <= target + upper_tolerance;
    }

    IntervalEstimate two_collision_tread_estimate(
        float phase_stride_guard_cap,
        float min_width) const
    {
        IntervalEstimate est;
        if (!cfg_.ratchet_two_collision_enabled || cfg_.memory_len < 2 ||
            !toe_valid_[0]) {
            return est;
        }

        const auto& latest_toe = toe_marks_[0];
        const bool latest_valid =
            latest_toe[4] >= cfg_.ratchet_collision_min_confidence;
        const float latest_age = latest_toe[5];
        const float latest_z = latest_toe[8];
        const float latest_s = latest_toe[9];
        if (!latest_valid) {
            return est;
        }

        for (int older_idx = 1; older_idx < cfg_.memory_len; ++older_idx) {
            if (!toe_valid_[older_idx]) {
                continue;
            }
            const auto& older_toe = toe_marks_[older_idx];
            const bool older_valid =
                older_toe[4] >= cfg_.ratchet_collision_min_confidence &&
                older_toe[5] > latest_age + 1.0e-6f;
            if (!older_valid) {
                continue;
            }
            const float toe_delta_z = latest_z - older_toe[8];
            const float toe_delta_s = latest_s - older_toe[9];
            float step_count = 0.0f;
            for (int footprint_idx = 0; footprint_idx < cfg_.memory_len - 1; ++footprint_idx) {
                if (!footprint_valid_[footprint_idx] ||
                    !footprint_valid_[footprint_idx + 1]) {
                    continue;
                }
                const auto& newer = footprints_[footprint_idx];
                const auto& older = footprints_[footprint_idx + 1];
                const float pair_age = 0.5f * (newer[6] + older[6]);
                const float pair_stride = newer[10] - older[10];
                const float pair_height = newer[9] - older[9];
                const bool between_toe_hits =
                    pair_age < older_toe[5] - 1.0e-6f &&
                    pair_age > latest_age + 1.0e-6f;
                const bool forward_up_pair =
                    pair_stride >= cfg_.ratchet_min_stride_m &&
                    pair_height > cfg_.ratchet_height_threshold_m;
                if (between_toe_hits && forward_up_pair) {
                    step_count += 1.0f;
                }
            }

            const float riser_ref = ratchet_.validated_riser_height_s >
                    cfg_.ratchet_height_threshold_m
                ? ratchet_.validated_riser_height_s
                : cfg_.ratchet_two_collision_nominal_riser_height_m;
            const float height_layer_delta = std::max(
                1.0f,
                std::round(toe_delta_z / std::max(riser_ref, 1.0e-6f)));
            const float height_error =
                std::abs(toe_delta_z - height_layer_delta * riser_ref);
            const float height_tolerance = std::min(
                cfg_.ratchet_two_collision_height_layer_tolerance_m,
                std::max(0.025f, 0.5f * riser_ref));
            const bool height_layer_valid = height_error <= height_tolerance;
            const float layer_delta_candidate =
                cfg_.ratchet_two_collision_use_height_layers
                    ? height_layer_delta
                    : step_count;
            const float candidate_tread =
                toe_delta_s / std::max(layer_delta_candidate, 1.0f);
            const float min_layer_delta =
                static_cast<float>(cfg_.ratchet_two_collision_min_layer_delta);
            bool step_count_gate = step_count >= min_layer_delta;
            if (!cfg_.ratchet_second_collision_requires_up_step) {
                step_count_gate = true;
            }
            bool layer_delta_gate = layer_delta_candidate >= min_layer_delta;
            if (cfg_.ratchet_two_collision_use_height_layers) {
                layer_delta_gate = layer_delta_gate && height_layer_valid;
            }
            const bool candidate =
                toe_delta_z >= cfg_.ratchet_two_collision_min_height_delta_m &&
                toe_delta_s >= cfg_.ratchet_min_stride_m &&
                step_count_gate &&
                layer_delta_gate &&
                candidate_tread >= cfg_.ratchet_two_collision_tread_min_m &&
                candidate_tread <= cfg_.ratchet_two_collision_tread_max_m;
            if (!candidate) {
                continue;
            }

            est = make_interval(
                candidate_tread * cfg_.ratchet_two_collision_stride_layers,
                phase_stride_guard_cap,
                min_width);
            est.tread_depth = candidate_tread;
            est.layer_delta = layer_delta_candidate;
            return est;
        }
        return est;
    }

    AnchorGeometry anchor_collision_geometry(
        const ToeFeatures& toe,
        bool old_anchor_active,
        bool post_first_unconfirmed,
        bool toe_hit_context_candidate,
        bool toe_confident,
        float phase_stride_guard_cap,
        float min_width) const
    {
        AnchorGeometry out;
        if (!cfg_.ratchet_anchor_collision_enabled || !old_anchor_active) {
            return out;
        }
        const float anchor_delta_s = toe[4] - ratchet_.collision_anchor_s;
        const float anchor_delta_z = toe[5] - ratchet_.collision_anchor_z;
        const float riser_ref = ratchet_.validated_riser_height_s >
                cfg_.ratchet_height_threshold_m
            ? ratchet_.validated_riser_height_s
            : cfg_.ratchet_two_collision_nominal_riser_height_m;
        const float height_layer_delta = std::max(
            1.0f,
            std::round(anchor_delta_z / std::max(riser_ref, 1.0e-6f)));
        const float height_error =
            std::abs(anchor_delta_z - height_layer_delta * riser_ref);
        const float height_tolerance = std::min(
            cfg_.ratchet_two_collision_height_layer_tolerance_m,
            std::max(0.025f, 0.5f * riser_ref));
        const bool height_layer_valid = height_error <= height_tolerance;
        const float layer_delta = cfg_.ratchet_two_collision_use_height_layers
            ? height_layer_delta
            : static_cast<float>(std::max(1, ratchet_.collision_anchor_up_steps));
        const float min_layer_delta =
            static_cast<float>(cfg_.ratchet_two_collision_min_layer_delta);
        bool anchor_step_gate =
            static_cast<float>(ratchet_.collision_anchor_up_steps) >= min_layer_delta;
        if (!cfg_.ratchet_second_collision_requires_up_step) {
            anchor_step_gate = true;
        }
        bool anchor_layer_gate = layer_delta >= min_layer_delta;
        if (cfg_.ratchet_two_collision_use_height_layers) {
            anchor_layer_gate = anchor_layer_gate && height_layer_valid;
        }
        const bool anchor_height_gate =
            anchor_delta_z >= cfg_.ratchet_two_collision_min_height_delta_m;
        out.second_riser_gate =
            anchor_step_gate && anchor_height_gate && anchor_layer_gate;
        out.second_hit_gate =
            post_first_unconfirmed &&
            old_anchor_active &&
            toe_hit_context_candidate &&
            toe_confident &&
            out.second_riser_gate &&
            anchor_delta_s >= cfg_.ratchet_min_stride_m;

        const float anchor_tread =
            anchor_delta_s / std::max(layer_delta, 1.0f);
        out.collision_valid =
            toe_hit_context_candidate &&
            toe_confident &&
            anchor_delta_s >= cfg_.ratchet_min_stride_m &&
            anchor_step_gate &&
            anchor_layer_gate &&
            anchor_height_gate &&
            anchor_tread >= cfg_.ratchet_two_collision_tread_min_m &&
            anchor_tread <= cfg_.ratchet_two_collision_tread_max_m;
        if (out.collision_valid) {
            out.collision = make_interval(
                anchor_tread * cfg_.ratchet_two_collision_stride_layers,
                phase_stride_guard_cap,
                min_width);
            out.collision.tread_depth = anchor_tread;
            out.collision.layer_delta = layer_delta;
            out.collision_valid = out.collision.valid;
        }

        out.fallback = out.second_hit_gate && !out.collision_valid;
        if (out.fallback) {
            const float fallback_tread = clamp(
                anchor_tread,
                cfg_.ratchet_two_collision_tread_min_m,
                cfg_.ratchet_two_collision_tread_max_m);
            out.fallback_interval = make_interval(
                fallback_tread * cfg_.ratchet_two_collision_stride_layers,
                phase_stride_guard_cap,
                min_width);
            out.fallback_interval.tread_depth = fallback_tread;
            out.fallback_interval.layer_delta = layer_delta;
        }
        return out;
    }

    void update_ratchet(
        const std::array<PairFeatures, kPairCount>& pairs,
        const std::array<SameFootFeatures, 6>& same_steps,
        const ToeFeatures& toe,
        bool new_footprint_any,
        bool new_toe_mark_any,
        float dt)
    {
        const float aged = ratchet_.active ? ratchet_.age_s + dt : ratchet_.age_s;
        const bool latest_valid = pairs[0][0] > 0.5f;
        const float latest_stride = pairs[0][5];
        const float latest_height = pairs[0][6];
        const bool latest_same_valid = same_steps[0][0] > 0.5f;
        const float latest_same_stride = same_steps[0][2];
        const float latest_same_height = same_steps[0][3];
        const bool bootstrap_reference_valid =
            same_steps[1][0] > 0.5f &&
            same_steps[1][1] >= cfg_.ratchet_sequence_min_confidence &&
            same_steps[1][2] >= cfg_.ratchet_min_stride_m &&
            std::abs(same_steps[1][3]) <= cfg_.ratchet_flat_height_threshold_m;
        const float bootstrap_reference_stride = same_steps[1][2];
        const bool raw_step_valid = latest_same_valid || latest_valid;
        const float raw_step_stride = latest_same_valid ? latest_same_stride : latest_stride;
        const float raw_step_height = latest_same_valid ? latest_same_height : latest_height;
        const SequenceFeatures seq = latest_sequence_features();
        const float step_score = seq.score;
        const float step_stride = seq.same_foot_stride;
        const float step_height = seq.same_foot_rise;
        const bool raw_forward_up_step =
            new_footprint_any && raw_step_valid &&
            raw_step_stride >= cfg_.ratchet_min_stride_m &&
            raw_step_height > cfg_.ratchet_height_threshold_m;
        const bool forward_up_step = new_footprint_any && seq.valid;
        const bool flat_step =
            new_footprint_any && latest_valid &&
            std::abs(latest_height) <= cfg_.ratchet_flat_height_threshold_m;
        const bool toe_mark_candidate = new_toe_mark_any && toe[0] > 0.5f;
        const bool toe_relation_valid = toe[6] > 0.5f;
        const float toe_delta_s = std::max(toe[7], 0.0f);
        const bool toe_relation_candidate =
            toe_mark_candidate && toe_relation_valid && toe_delta_s >= cfg_.ratchet_min_stride_m;
        const bool toe_context = cfg_.ratchet_first_collision_enters_stair_mode
            ? (ratchet_.active || forward_up_step || toe_relation_candidate)
            : (ratchet_.active || forward_up_step);
        const bool toe_context_candidate = toe_relation_candidate && toe_context;
        const bool toe_hit_context_candidate =
            toe_mark_candidate &&
            (ratchet_.active || forward_up_step || toe_context_candidate);
        const float toe_confidence = toe[1];

        const float max_stride = cfg_.ratchet_max_stride_m;
        const float min_width = std::min(
            cfg_.ratchet_min_interval_width_m,
            std::max(cfg_.ratchet_max_stride_m - cfg_.ratchet_min_stride_m, 1.0e-6f));
        const float raw_stride_evidence =
            forward_up_step ? std::max(step_stride, 0.0f) : 0.0f;
        const bool guard_context =
            latest_valid &&
            latest_stride >= cfg_.ratchet_min_stride_m &&
            latest_height > cfg_.ratchet_height_threshold_m;
        const float stride_guard_cap = clamp(
            guard_context
                ? cfg_.ratchet_same_foot_stride_guard_layers *
                          std::max(latest_stride, 0.0f) +
                      cfg_.ratchet_same_foot_stride_guard_margin_m
                : max_stride,
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);

        const float old_lower = ratchet_.active ? ratchet_.lower_s : 0.0f;
        const float old_stride_lower =
            ratchet_.active ? ratchet_.same_foot_stride_lower_s : 0.0f;
        const float old_probe = ratchet_.active ? ratchet_.probe_target_s : 0.0f;
        const bool old_interval_confirmed =
            ratchet_.active && ratchet_.interval_confirmed;
        const bool old_first_collision_seen =
            ratchet_.active && ratchet_.first_collision_seen && !old_interval_confirmed;
        const bool old_anchor_active =
            ratchet_.active && ratchet_.collision_anchor_active && !old_interval_confirmed;
        const bool post_first_collision_context =
            old_first_collision_seen || old_anchor_active;
        const bool wide_cap_context =
            post_first_collision_context && !old_interval_confirmed;
        const bool old_probe_bootstrap_ready =
            ratchet_.active &&
            ratchet_.probe_bootstrap_ready &&
            post_first_collision_context;
        const bool old_probe_bootstrap_provisional =
            old_probe_bootstrap_ready && ratchet_.probe_bootstrap_provisional;
        const float old_validated_stride =
            old_probe_bootstrap_ready ? ratchet_.validated_stride_s : 0.0f;
        const float old_validated_riser_height =
            ratchet_.active ? ratchet_.validated_riser_height_s : 0.0f;
        const float phase_stride_guard_cap =
            wide_cap_context ? max_stride : stride_guard_cap;
        const bool first_layer_touchdown =
            post_first_collision_context &&
            raw_forward_up_step &&
            old_anchor_active &&
            ratchet_.collision_anchor_up_steps < 1;
        const bool first_layer_bootstrap =
            first_layer_touchdown && bootstrap_reference_valid;
        const bool fallback_sequence_bootstrap =
            post_first_collision_context &&
            forward_up_step &&
            !old_probe_bootstrap_ready &&
            !first_layer_touchdown;
        const bool provisional_sequence_candidate =
            post_first_collision_context &&
            forward_up_step &&
            old_probe_bootstrap_provisional;
        const bool bootstrap_event =
            first_layer_bootstrap || fallback_sequence_bootstrap;
        float new_validated_riser_height = old_validated_riser_height;
        if (first_layer_touchdown) {
            new_validated_riser_height = raw_step_height;
        } else if (
            fallback_sequence_bootstrap &&
            old_validated_riser_height <= cfg_.ratchet_height_threshold_m) {
            new_validated_riser_height = 0.5f * (seq.first_rise + seq.second_rise);
        }
        new_validated_riser_height = clamp(
            new_validated_riser_height,
            0.0f,
            cfg_.ratchet_sequence_max_adjacent_height_m);
        const float bootstrap_stride = clamp(
            first_layer_bootstrap ? bootstrap_reference_stride : step_stride,
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_probe_cap_m);
        const bool lower_evidence_step = forward_up_step;
        const float stride_evidence = lower_evidence_step
            ? std::min(std::max(step_stride, 0.0f), phase_stride_guard_cap)
            : 0.0f;
        const float toe_confidence_threshold = post_first_collision_context
            ? cfg_.ratchet_post_first_collision_collision_min_confidence
            : cfg_.ratchet_collision_min_confidence;
        const bool toe_confident = toe_confidence >= toe_confidence_threshold;
        const bool old_soft_upper_active =
            ratchet_.active && ratchet_.soft_upper_active && !old_interval_confirmed;
        Mode old_mode = ratchet_.active ? ratchet_.mode : Mode::Probe;
        if (old_interval_confirmed && old_mode == Mode::Probe) {
            old_mode = Mode::Lock;
        }
        const bool old_recovery_pending =
            ratchet_.active && ratchet_.recovery_pending;
        const bool old_recovery_attempt_seen =
            old_recovery_pending && ratchet_.recovery_attempt_seen;
        const bool old_recovery_reduction_seen =
            old_recovery_pending && ratchet_.recovery_reduction_seen;
        (void)old_recovery_reduction_seen;
        const float previous_actual_stride = ratchet_.last_forward_up_stride;
        const float old_recovery_target =
            old_recovery_pending ? ratchet_.recovery_target_s : 0.0f;
        const bool old_lock_target_valid =
            ratchet_.active &&
            old_interval_confirmed &&
            ratchet_.lock_target_s > cfg_.ratchet_min_stride_m;
        const float old_lock_target =
            old_lock_target_valid ? ratchet_.lock_target_s : 0.0f;

        const bool lower_ratchet_enabled = old_mode == Mode::Probe;
        const float old_upper =
            old_interval_confirmed ? ratchet_.upper_s : max_stride;
        const float old_stride_upper =
            old_interval_confirmed ? ratchet_.same_foot_stride_upper_s : max_stride;
        const float old_soft_upper =
            old_soft_upper_active ? ratchet_.upper_s : max_stride;
        const float old_soft_stride_upper =
            old_soft_upper_active ? ratchet_.same_foot_stride_upper_s : max_stride;
        const float lower_target = clamp(
            stride_evidence + cfg_.ratchet_no_hit_lower_margin_m,
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);
        const float lower_probe_increment = old_first_collision_seen
            ? cfg_.ratchet_post_first_collision_probe_increment_m
            : cfg_.ratchet_probe_increment_m;
        const float lower_growth_cap =
            ratchet_.active ? old_lower + lower_probe_increment : lower_target;
        const float stride_lower_growth_cap =
            ratchet_.active ? old_stride_lower + lower_probe_increment : lower_target;
        const float lower_upper_cap = clamp(
            old_interval_confirmed ? old_stride_upper - min_width : phase_stride_guard_cap,
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);
        const float lower_target_lag_cap = clamp(
            ratchet_.active ? old_probe - cfg_.ratchet_lower_target_lag_margin_m
                            : lower_target,
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);
        const float lower_hard_cap =
            std::min(std::min(lower_upper_cap, phase_stride_guard_cap), lower_target_lag_cap);
        const float lower_candidate =
            std::min(std::min(lower_target, lower_growth_cap), lower_hard_cap);
        const float stride_lower_candidate =
            std::min(std::min(lower_target, stride_lower_growth_cap), lower_hard_cap);
        float new_lower = ratchet_.lower_s;
        float new_stride_lower = ratchet_.same_foot_stride_lower_s;
        if (lower_evidence_step && lower_ratchet_enabled) {
            new_lower = clamp(
                std::max(old_lower, lower_candidate),
                cfg_.ratchet_min_stride_m,
                cfg_.ratchet_max_stride_m);
            new_stride_lower = clamp(
                std::max(old_stride_lower, stride_lower_candidate),
                cfg_.ratchet_min_stride_m,
                cfg_.ratchet_max_stride_m);
        }
        const bool lower_updated =
            lower_evidence_step &&
            lower_ratchet_enabled &&
            (new_lower > old_lower + 1.0e-5f ||
             new_stride_lower > old_stride_lower + 1.0e-5f);
        (void)lower_updated;

        const float collision_upper_raw = clamp(
            toe_delta_s - cfg_.ratchet_toe_anchor_offset_m -
                cfg_.ratchet_collision_margin_m,
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);
        const float collision_upper =
            std::min(collision_upper_raw, phase_stride_guard_cap);
        const IntervalEstimate two_collision =
            two_collision_tread_estimate(phase_stride_guard_cap, min_width);
        const bool two_collision_candidate =
            toe_hit_context_candidate && toe_confident && two_collision.valid;
        const AnchorGeometry anchor = anchor_collision_geometry(
            toe,
            old_anchor_active,
            post_first_collision_context && !old_interval_confirmed,
            toe_hit_context_candidate,
            toe_confident,
            phase_stride_guard_cap,
            min_width);

        const bool anchor_second_hit_fallback = anchor.fallback;
        const bool geometry_collision_candidate =
            two_collision_candidate || anchor.collision_valid || anchor_second_hit_fallback;
        const bool both_geometry_candidates =
            two_collision_candidate && anchor.collision_valid;
        IntervalEstimate geometry = anchor_second_hit_fallback
            ? anchor.fallback_interval
            : (anchor.collision_valid ? anchor.collision : two_collision);
        if (both_geometry_candidates) {
            geometry.lower = std::max(two_collision.lower, anchor.collision.lower);
            geometry.upper = std::min(two_collision.upper, anchor.collision.upper);
            geometry.center = 0.5f * (geometry.lower + geometry.upper);
            geometry.valid = geometry.upper >= geometry.lower + min_width;
        }
        const bool geometry_bounds_valid =
            geometry.valid && geometry.upper >= geometry.lower + min_width;
        const float lower_cross_reference = std::max(new_lower, new_stride_lower);
        const bool lower_cross_consistent =
            lower_cross_reference <=
            geometry.center + cfg_.ratchet_two_collision_lower_cross_margin_m;
        const bool upper_cross_consistent =
            collision_upper + cfg_.ratchet_two_collision_upper_cross_margin_m >=
            geometry.center;
        const bool measured_collision_upper_valid =
            toe_relation_candidate &&
            collision_upper >= cfg_.ratchet_min_stride_m + min_width;
        const bool lower_cross_informative =
            lower_cross_reference >
            cfg_.ratchet_min_stride_m + cfg_.ratchet_two_collision_lower_cross_margin_m;
        const bool geometry_cross_check_consistent =
            lower_cross_consistent &&
            (lower_cross_informative
                 ? true
                 : (measured_collision_upper_valid ? upper_cross_consistent : true));
        const bool post_first_unconfirmed =
            post_first_collision_context && !old_interval_confirmed;
        const float geometry_lower_cross = post_first_unconfirmed
            ? geometry.lower
            : std::max(new_lower, geometry.lower);
        const float geometry_stride_lower_cross = post_first_unconfirmed
            ? geometry.lower
            : std::max(new_stride_lower, geometry.lower);
        const bool cross_checked_geometry_interval_evidence =
            geometry_collision_candidate &&
            geometry_bounds_valid &&
            geometry_cross_check_consistent &&
            (post_first_unconfirmed ||
             (old_interval_confirmed &&
              geometry.upper >= geometry_lower_cross + min_width &&
              geometry.upper >= geometry_stride_lower_cross + min_width));
        const bool raw_rejected_second_hit =
            post_first_unconfirmed &&
            geometry_collision_candidate &&
            geometry_bounds_valid &&
            !geometry_cross_check_consistent;
        const int old_rejected_count =
            post_first_unconfirmed ? ratchet_.rejected_second_hit_count : 0;
        const float old_rejected_target =
            old_rejected_count > 0 ? ratchet_.rejected_second_hit_target_s : 0.0f;
        const bool rejected_second_hit_consistent =
            old_rejected_count > 0 &&
            std::abs(geometry.center - old_rejected_target) <=
                cfg_.ratchet_rejected_second_hit_target_tolerance_m;
        int rejected_second_hit_count = old_rejected_count;
        float rejected_second_hit_target = old_rejected_target;
        if (raw_rejected_second_hit) {
            rejected_second_hit_count = rejected_second_hit_consistent
                ? old_rejected_count + 1
                : 1;
            rejected_second_hit_target = geometry.center;
        }
        const bool rejected_second_hit_promoted =
            raw_rejected_second_hit &&
            rejected_second_hit_count >= cfg_.ratchet_rejected_second_hit_confirm_count;
        const bool phase_second_hit_evidence =
            post_first_unconfirmed &&
            geometry_collision_candidate &&
            geometry_bounds_valid &&
            (two_collision_candidate || anchor.second_hit_gate);
        const bool geometry_interval_evidence =
            cross_checked_geometry_interval_evidence ||
            rejected_second_hit_promoted ||
            anchor_second_hit_fallback ||
            phase_second_hit_evidence;
        const bool two_cross_check_rejected =
            two_collision_candidate && !geometry_cross_check_consistent;
        const bool anchor_cross_check_rejected =
            anchor.collision_valid && !geometry_cross_check_consistent;
        const bool second_riser_gate_pass =
            two_collision_candidate ||
            (old_anchor_active &&
             toe_hit_context_candidate &&
             toe_confident &&
             anchor.second_riser_gate);
        const bool conservative_second_hit =
            raw_rejected_second_hit && !rejected_second_hit_promoted;
        float second_hit_upper_bound =
            measured_collision_upper_valid ? collision_upper : geometry.upper;
        const bool fallback_measured_upper_valid =
            measured_collision_upper_valid && collision_upper >= anchor.fallback_interval.lower;
        const float fallback_second_hit_upper = fallback_measured_upper_valid
            ? std::min(collision_upper, anchor.fallback_interval.upper)
            : anchor.fallback_interval.upper;
        if (anchor_second_hit_fallback) {
            second_hit_upper_bound = fallback_second_hit_upper;
        }
        const bool duplicate_first_riser_collision =
            old_first_collision_seen &&
            toe_hit_context_candidate &&
            toe_confident &&
            !old_interval_confirmed &&
            !second_riser_gate_pass;
        const bool single_collision_interval_ok =
            toe_context_candidate &&
            toe_confident &&
            collision_upper >= new_lower + min_width &&
            collision_upper >= new_stride_lower + min_width;
        const bool single_interval_evidence =
            cfg_.ratchet_single_collision_confirms_interval
                ? single_collision_interval_ok
                : (single_collision_interval_ok && old_interval_confirmed);
        const bool toe_evidence =
            single_interval_evidence || geometry_interval_evidence;
        bool first_collision_signal =
            toe_context_candidate &&
            toe_confident &&
            !old_interval_confirmed &&
            !post_first_unconfirmed &&
            !single_interval_evidence &&
            !duplicate_first_riser_collision;
        if (!cfg_.ratchet_first_collision_enters_stair_mode) {
            first_collision_signal = false;
        }
        if (first_collision_signal && !old_interval_confirmed) {
            new_lower = std::max(new_lower, cfg_.ratchet_min_stride_m);
            new_stride_lower = std::max(new_stride_lower, cfg_.ratchet_min_stride_m);
        }
        const bool soft_upper_allowed =
            cfg_.ratchet_single_collision_confirms_interval ||
            old_interval_confirmed ||
            two_cross_check_rejected ||
            anchor_cross_check_rejected;
        bool soft_upper_evidence =
            toe_context_candidate &&
            toe_confident &&
            soft_upper_allowed &&
            !post_first_unconfirmed &&
            !(toe_evidence || first_collision_signal) &&
            !duplicate_first_riser_collision;
        soft_upper_evidence = soft_upper_evidence || conservative_second_hit;
        const bool has_evidence =
            forward_up_step ||
            first_layer_touchdown ||
            toe_evidence ||
            soft_upper_evidence ||
            first_collision_signal ||
            duplicate_first_riser_collision;

        if (geometry_interval_evidence) {
            new_lower = geometry_lower_cross;
            new_stride_lower = geometry_stride_lower_cross;
        }
        const float conservative_lower_cap = clamp(
            std::min(geometry.lower, second_hit_upper_bound - min_width),
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);
        if (conservative_second_hit) {
            new_lower = std::min(new_lower, conservative_lower_cap);
            new_stride_lower = std::min(new_stride_lower, conservative_lower_cap);
        }

        const bool confirmed_soft_upper_evidence =
            soft_upper_evidence && old_interval_confirmed;
        float new_upper_candidate = old_upper;
        if ((toe_evidence && !geometry_interval_evidence) ||
            confirmed_soft_upper_evidence) {
            new_upper_candidate = std::min(old_upper, collision_upper);
        }
        if (conservative_second_hit) {
            new_upper_candidate = std::min(old_upper, second_hit_upper_bound);
        }
        if (geometry_interval_evidence) {
            new_upper_candidate = geometry.upper;
        }
        const bool measured_confirmed_collision =
            old_interval_confirmed &&
            measured_collision_upper_valid &&
            (toe_evidence || soft_upper_evidence);
        if (measured_confirmed_collision) {
            new_upper_candidate = std::min(new_upper_candidate, collision_upper);
        }
        float new_stride_upper_candidate = old_stride_upper;
        if ((toe_evidence && !geometry_interval_evidence) ||
            confirmed_soft_upper_evidence) {
            new_stride_upper_candidate = std::min(old_stride_upper, collision_upper);
        }
        if (conservative_second_hit) {
            new_stride_upper_candidate = std::min(old_stride_upper, second_hit_upper_bound);
        }
        if (geometry_interval_evidence) {
            new_stride_upper_candidate = geometry.upper;
        }
        if (measured_confirmed_collision) {
            new_stride_upper_candidate =
                std::min(new_stride_upper_candidate, collision_upper);
        }
        float new_soft_upper_candidate = old_soft_upper;
        float new_soft_stride_upper_candidate = old_soft_stride_upper;
        if (soft_upper_evidence) {
            new_soft_upper_candidate = std::min(old_soft_upper, collision_upper);
            new_soft_stride_upper_candidate =
                std::min(old_soft_stride_upper, collision_upper);
        }
        if (conservative_second_hit) {
            new_soft_upper_candidate =
                std::min(old_soft_upper, second_hit_upper_bound);
            new_soft_stride_upper_candidate =
                std::min(old_soft_stride_upper, second_hit_upper_bound);
        }
        const bool new_interval_confirmed =
            old_interval_confirmed ||
            (toe_evidence &&
             std::isfinite(new_upper_candidate) &&
             new_upper_candidate > cfg_.ratchet_min_stride_m);
        if (new_interval_confirmed) {
            new_lower = std::min(
                new_lower,
                std::max(new_upper_candidate - min_width, cfg_.ratchet_min_stride_m));
            new_stride_lower = std::min(
                new_stride_lower,
                std::max(
                    new_stride_upper_candidate - min_width,
                    cfg_.ratchet_min_stride_m));
        }

        const bool soft_upper_candidate_active =
            (old_soft_upper_active || soft_upper_evidence) && !new_interval_confirmed;
        const int ttl_full = cfg_.ratchet_soft_upper_ttl_steps;
        const int old_soft_ttl =
            old_soft_upper_active ? ratchet_.soft_upper_ttl_remaining : 0;
        const bool safe_forward_no_hit =
            forward_up_step && !(toe_evidence || soft_upper_evidence);
        int new_soft_ttl = soft_upper_evidence ? ttl_full : old_soft_ttl;
        if (old_soft_upper_active && safe_forward_no_hit && !soft_upper_evidence) {
            new_soft_ttl = std::max(0, new_soft_ttl - 1);
        }
        if (new_interval_confirmed) {
            new_soft_ttl = 0;
        }
        const bool new_soft_upper_active =
            soft_upper_candidate_active && new_soft_ttl > 0;
        const bool soft_upper_released =
            old_soft_upper_active &&
            !soft_upper_evidence &&
            safe_forward_no_hit &&
            !new_soft_upper_active;
        float new_upper = 0.0f;
        float new_stride_upper = 0.0f;
        if (new_interval_confirmed) {
            new_upper = new_upper_candidate;
            new_stride_upper = new_stride_upper_candidate;
        } else if (new_soft_upper_active) {
            new_upper = new_soft_upper_candidate;
            new_stride_upper = new_soft_stride_upper_candidate;
        }
        new_upper = clamp(new_upper, 0.0f, cfg_.ratchet_max_stride_m);
        new_stride_upper = clamp(new_stride_upper, 0.0f, cfg_.ratchet_max_stride_m);

        float new_collision_upper = ratchet_.collision_upper_s;
        if (geometry_interval_evidence) {
            new_collision_upper = geometry.upper;
        } else if (toe_evidence || soft_upper_evidence) {
            new_collision_upper = collision_upper;
        }
        if (conservative_second_hit) {
            new_collision_upper = second_hit_upper_bound;
        }
        if (measured_confirmed_collision) {
            new_collision_upper = collision_upper;
        }

        const bool collision_update = toe_evidence || soft_upper_evidence;
        const bool post_first_collision_probe =
            (old_first_collision_seen || first_collision_signal) && !new_interval_confirmed;
        const bool valid_second_hit_event =
            post_first_collision_context &&
            !old_interval_confirmed &&
            geometry_interval_evidence;
        const bool phase_collision_update =
            valid_second_hit_event || (old_interval_confirmed && collision_update);
        Mode new_mode = phase_collision_update ? Mode::Backoff : old_mode;
        if (soft_upper_released &&
            !new_interval_confirmed &&
            old_mode == Mode::Backoff &&
            !collision_update) {
            new_mode = Mode::Probe;
        }
        const bool open_probe_latch =
            post_first_collision_probe &&
            !new_interval_confirmed &&
            !new_soft_upper_active &&
            !phase_collision_update;
        if (open_probe_latch) {
            new_mode = Mode::Probe;
        }

        const float prior_probe_lower = std::max(old_lower, old_stride_lower);
        const float collision_target_upper = clamp(
            second_hit_upper_bound - cfg_.ratchet_lock_probe_lower_margin_m,
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);
        const float collision_target_lower = clamp(
            prior_probe_lower + cfg_.ratchet_lock_probe_lower_margin_m,
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);
        const bool collision_bracket_valid =
            valid_second_hit_event && collision_target_upper >= collision_target_lower;
        const float bracketed_geometry_target = std::max(
            collision_target_lower,
            std::min(geometry.center, collision_target_upper));
        const float fused_lock_target = clamp(
            collision_bracket_valid
                ? bracketed_geometry_target
                : std::min(geometry.center, collision_target_upper),
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);
        const float fused_interval_lower_cap = std::min(
            fused_lock_target - cfg_.ratchet_lock_probe_lower_margin_m,
            second_hit_upper_bound - min_width);
        const float fused_interval_lower = clamp(
            std::min(
                std::max(prior_probe_lower, geometry.lower),
                fused_interval_lower_cap),
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);
        if (valid_second_hit_event) {
            new_lower = fused_interval_lower;
            new_stride_lower = fused_interval_lower;
            new_upper = second_hit_upper_bound;
            new_stride_upper = second_hit_upper_bound;
            new_collision_upper = second_hit_upper_bound;
        }

        float persistent_lock_target = 0.5f * (new_stride_lower + new_stride_upper);
        const float persistent_lock_upper = clamp(
            new_stride_upper - cfg_.ratchet_lock_probe_lower_margin_m,
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);
        const float persistent_lock_lower = std::min(
            clamp(
                new_stride_lower + cfg_.ratchet_lock_probe_lower_margin_m,
                cfg_.ratchet_min_stride_m,
                cfg_.ratchet_max_stride_m),
            persistent_lock_upper);
        persistent_lock_target = std::max(
            persistent_lock_lower,
            std::min(persistent_lock_target, persistent_lock_upper));
        if (old_lock_target_valid) {
            persistent_lock_target = old_lock_target;
        }
        const float new_lock_target = clamp(
            valid_second_hit_event ? fused_lock_target : persistent_lock_target,
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);
        const float lock_lower_candidate =
            std::min(new_stride_lower + cfg_.ratchet_lock_margin_m, new_lock_target);
        const float lock_upper_candidate =
            std::max(new_stride_upper - cfg_.ratchet_lock_margin_m, new_lock_target);
        const bool lock_interval_valid =
            new_interval_confirmed && lock_upper_candidate >= lock_lower_candidate;
        const bool lock_collision_reopen =
            collision_update && old_mode == Mode::Lock;
        bool new_upper_seen =
            (ratchet_.active && ratchet_.upper_seen) || collision_update;
        if (soft_upper_released && !new_interval_confirmed) {
            new_upper_seen = false;
        }

        const float bootstrap_target = first_layer_bootstrap
            ? std::max(
                  std::min(
                      bootstrap_stride + cfg_.ratchet_probe_bootstrap_increment_m,
                      cfg_.ratchet_probe_cap_m),
                  std::min(
                      cfg_.ratchet_probe_provisional_min_target_m,
                      cfg_.ratchet_probe_cap_m))
            : std::min(
                  bootstrap_stride + cfg_.ratchet_probe_bootstrap_increment_m,
                  cfg_.ratchet_probe_cap_m);
        const bool provisional_sequence_rebase =
            provisional_sequence_candidate && !collision_update;
        const float provisional_rebase_target = std::min(
            step_stride + cfg_.ratchet_probe_bootstrap_increment_m,
            cfg_.ratchet_probe_cap_m);
        const bool probe_touchdown =
            old_probe_bootstrap_ready &&
            !old_probe_bootstrap_provisional &&
            post_first_collision_probe &&
            forward_up_step &&
            !collision_update;
        const bool previous_probe_stride_valid =
            previous_actual_stride > cfg_.ratchet_min_stride_m;
        const float probe_growth = step_stride - previous_actual_stride;
        const bool probe_completion =
            probe_touchdown && previous_probe_stride_valid && probe_growth > 1.0e-4f;
        const float rolling_probe_target = std::min(
            std::min(
                std::max(old_probe, step_stride + cfg_.ratchet_post_first_collision_probe_increment_m),
                old_probe + cfg_.ratchet_post_first_collision_probe_increment_m),
            cfg_.ratchet_probe_cap_m);
        float phase_probe_target = old_probe;
        if (provisional_sequence_rebase) {
            phase_probe_target = provisional_rebase_target;
        } else if (bootstrap_event) {
            phase_probe_target = bootstrap_target;
        } else if (probe_touchdown) {
            phase_probe_target = rolling_probe_target;
        }
        if (first_collision_signal && !bootstrap_event) {
            phase_probe_target = ratchet_.active ? old_probe : cfg_.ratchet_min_stride_m;
        }

        float new_validated_stride = old_validated_stride;
        if (provisional_sequence_rebase) {
            new_validated_stride = step_stride;
        } else if (bootstrap_event) {
            new_validated_stride = bootstrap_stride;
        } else if (probe_touchdown) {
            new_validated_stride = step_stride;
        }
        const bool upper_target_valid =
            new_interval_confirmed || new_soft_upper_active;
        const bool recovery_trigger =
            valid_second_hit_event || lock_collision_reopen;
        float triggered_recovery_target = std::max(
            lock_lower_candidate,
            new_lock_target - cfg_.ratchet_recovery_offset_m);
        triggered_recovery_target =
            std::min(triggered_recovery_target, new_lock_target);
        const float recovery_target =
            (recovery_trigger && !old_recovery_pending)
                ? triggered_recovery_target
                : old_recovery_target;
        const bool recovery_touchdown =
            old_recovery_pending && forward_up_step && !collision_update;
        const bool recovery_reduction_this_touchdown =
            recovery_touchdown &&
            step_stride <=
                previous_actual_stride - cfg_.ratchet_recovery_min_reduction_m;
        bool new_recovery_reduction_seen =
            recovery_trigger ? false : ratchet_.recovery_reduction_seen;
        new_recovery_reduction_seen =
            new_recovery_reduction_seen || recovery_reduction_this_touchdown;
        const bool recovery_completed =
            recovery_touchdown &&
            target_contains(
                step_stride,
                old_recovery_target,
                cfg_.ratchet_recovery_completion_lower_tolerance_m,
                cfg_.ratchet_recovery_completion_tolerance_m);
        bool new_recovery_pending =
            (old_recovery_pending || recovery_trigger) && !recovery_completed;
        bool new_recovery_attempt_seen =
            recovery_trigger ? false : old_recovery_attempt_seen;
        new_recovery_attempt_seen =
            (new_recovery_attempt_seen || recovery_touchdown) && new_recovery_pending;
        new_recovery_reduction_seen =
            new_recovery_reduction_seen && new_recovery_pending;
        if (new_recovery_pending) {
            new_mode = Mode::Backoff;
        }
        if (recovery_completed) {
            new_mode = new_interval_confirmed ? Mode::Lock : Mode::Probe;
        }

        const bool backoff_after_collision = new_mode == Mode::Backoff;
        const float backoff_or_hold_target =
            new_recovery_pending ? recovery_target : new_lock_target;
        const bool old_lock_bounds_valid =
            ratchet_.active && ratchet_.lock_upper_s > ratchet_.lock_lower_s;
        const float effective_lock_lower =
            old_lock_bounds_valid && old_mode == Mode::Lock
                ? ratchet_.lock_lower_s
                : lock_lower_candidate;
        const float effective_lock_upper =
            old_lock_bounds_valid && old_mode == Mode::Lock
                ? ratchet_.lock_upper_s
                : lock_upper_candidate;
        const float lock_center = std::max(
            effective_lock_lower,
            std::min(new_lock_target, effective_lock_upper));
        const bool actual_stride_inside_lock =
            forward_up_step &&
            lock_interval_valid &&
            step_stride >= lock_lower_candidate &&
            step_stride <= lock_upper_candidate &&
            !collision_update;
        const bool target_stride_inside_lock =
            cfg_.ratchet_lock_target_stable_enabled &&
            backoff_after_collision &&
            lock_interval_valid &&
            backoff_or_hold_target >= lock_lower_candidate &&
            backoff_or_hold_target <= lock_upper_candidate &&
            !collision_update;
        const bool lock_stable_evidence =
            (actual_stride_inside_lock || target_stride_inside_lock) &&
            backoff_after_collision;
        const int old_lock_stable_count =
            ratchet_.active ? ratchet_.lock_stable_count : 0;
        const bool stable_reset =
            collision_update ||
            (backoff_after_collision && forward_up_step && !lock_stable_evidence);
        int new_lock_stable_count = old_lock_stable_count;
        if (lock_stable_evidence) {
            new_lock_stable_count = old_lock_stable_count + 1;
        } else if (stable_reset) {
            new_lock_stable_count = 0;
        }
        const bool stable_lock_enter =
            backoff_after_collision &&
            !new_recovery_pending &&
            lock_interval_valid &&
            new_lock_stable_count >= cfg_.ratchet_lock_stable_steps;
        const bool lock_enter =
            (recovery_completed && new_interval_confirmed) || stable_lock_enter;
        if (lock_enter) {
            new_mode = Mode::Lock;
        }

        const bool previous_lock_active =
            ratchet_.active &&
            old_mode == Mode::Lock &&
            old_interval_confirmed &&
            !old_recovery_pending;
        const float old_phase_error =
            previous_lock_active ? ratchet_.lock_phase_error_s : 0.0f;
        const float actual_stride_error = step_stride - lock_center;
        float new_phase_error = old_phase_error;
        if (previous_lock_active &&
            forward_up_step &&
            lock_interval_valid &&
            !collision_update) {
            new_phase_error = old_phase_error + actual_stride_error;
        }
        if (lock_enter && (geometry_interval_evidence || recovery_completed)) {
            new_phase_error = cfg_.ratchet_lock_phase_initial_front_error_m;
        }
        const bool lock_phase_enabled =
            cfg_.ratchet_lock_phase_correction_enabled &&
            new_mode == Mode::Lock;
        if (lock_phase_enabled) {
            new_phase_error = clamp(
                new_phase_error,
                -cfg_.ratchet_lock_phase_max_error_m,
                cfg_.ratchet_lock_phase_max_error_m);
        } else {
            new_phase_error = 0.0f;
        }
        const float deadband = cfg_.ratchet_lock_phase_deadband_m;
        float lock_phase_correction = 0.0f;
        if (lock_phase_enabled) {
            if (new_phase_error > deadband) {
                lock_phase_correction =
                    (new_phase_error - deadband) *
                    cfg_.ratchet_lock_phase_correction_gain;
            } else if (new_phase_error < -deadband) {
                lock_phase_correction =
                    (new_phase_error + deadband) *
                    cfg_.ratchet_lock_phase_correction_gain;
            }
            lock_phase_correction = clamp(
                lock_phase_correction,
                -cfg_.ratchet_lock_phase_max_forward_m,
                cfg_.ratchet_lock_phase_max_backoff_m);
        }

        float new_probe = phase_probe_target;
        if (new_mode == Mode::Lock) {
            new_probe = new_lock_target;
        } else if (new_mode == Mode::Backoff) {
            new_probe = backoff_or_hold_target;
        }
        if (!(new_mode == Mode::Backoff || post_first_collision_probe)) {
            new_probe = std::max(new_probe, new_lower);
        }
        const float final_probe_cap =
            post_first_collision_probe && new_mode == Mode::Probe
                ? cfg_.ratchet_probe_cap_m
                : ((upper_target_valid || backoff_after_collision)
                       ? max_stride
                       : stride_guard_cap);
        new_probe = clamp(
            std::min(new_probe, final_probe_cap),
            cfg_.ratchet_min_stride_m,
            cfg_.ratchet_max_stride_m);

        const float new_last_stride = bootstrap_event
            ? bootstrap_stride
            : (forward_up_step ? std::max(step_stride, 0.0f)
                               : ratchet_.last_forward_up_stride);
        const float new_growth =
            (bootstrap_event || forward_up_step)
                ? new_last_stride - ratchet_.last_forward_up_stride
                : ratchet_.stride_growth;
        const float new_last_height =
            forward_up_step ? std::max(step_height, 0.0f)
                            : ratchet_.last_forward_up_height;
        const bool collision_or_signal =
            toe_evidence || soft_upper_evidence || first_collision_signal;
        int new_no_hit_steps = ratchet_.safe_no_hit_steps;
        if (forward_up_step && !collision_or_signal) {
            new_no_hit_steps += 1;
        }
        if (collision_or_signal) {
            new_no_hit_steps = 0;
        }
        float evidence_confidence = std::max(
            forward_up_step ? step_score : 0.0f,
            collision_or_signal ? clamp(toe_confidence, 0.0f, 1.0f) : 0.0f);
        if (new_interval_confirmed) {
            evidence_confidence = std::max(evidence_confidence, 0.75f);
        }
        const float new_confidence = has_evidence
            ? std::max(ratchet_.confidence, evidence_confidence)
            : ratchet_.confidence;
        int new_flat_steps = ratchet_.flat_pair_steps;
        if (flat_step && !has_evidence) {
            new_flat_steps += 1;
        }
        if (has_evidence) {
            new_flat_steps = 0;
        }
        const bool new_first_collision_seen =
            (old_first_collision_seen || first_collision_signal) &&
            !new_interval_confirmed;
        const int old_anchor_steps =
            old_anchor_active ? ratchet_.collision_anchor_up_steps : 0;
        const int anchor_step_increment =
            (old_anchor_active &&
             (first_layer_touchdown || forward_up_step) &&
             !collision_or_signal)
                ? 1
                : 0;
        int new_anchor_steps = old_anchor_steps + anchor_step_increment;
        float new_anchor_s =
            first_collision_signal ? toe[4] : ratchet_.collision_anchor_s;
        float new_anchor_z =
            first_collision_signal ? toe[5] : ratchet_.collision_anchor_z;
        if (first_collision_signal) {
            new_anchor_steps = 0;
        }
        const bool new_anchor_active =
            (old_anchor_active || first_collision_signal) && !new_interval_confirmed;
        const bool new_probe_bootstrap_ready =
            (old_probe_bootstrap_ready || bootstrap_event) && !new_interval_confirmed;
        bool new_probe_bootstrap_provisional = ratchet_.probe_bootstrap_provisional;
        if (first_layer_bootstrap) {
            new_probe_bootstrap_provisional = true;
        } else if (fallback_sequence_bootstrap || provisional_sequence_rebase) {
            new_probe_bootstrap_provisional = false;
        }
        new_probe_bootstrap_provisional =
            new_probe_bootstrap_provisional &&
            new_probe_bootstrap_ready &&
            !new_interval_confirmed;

        const bool stale =
            ratchet_.active && aged >= cfg_.age_norm_s && !has_evidence;
        const bool phase_latched =
            new_first_collision_seen || new_interval_confirmed;
        const bool flat_reset =
            ratchet_.active &&
            !phase_latched &&
            new_flat_steps >= cfg_.ratchet_reset_flat_pairs;
        const bool reset = stale || flat_reset;
        const bool active = (ratchet_.active || has_evidence) && !reset;
        if (!active) {
            ratchet_ = RatchetState{};
            return;
        }

        ratchet_.active = true;
        ratchet_.lower_s = new_lower;
        ratchet_.probe_target_s = new_probe;
        ratchet_.upper_s =
            (new_interval_confirmed || new_soft_upper_active) ? new_upper : 0.0f;
        ratchet_.collision_upper_s = new_collision_upper;
        ratchet_.same_foot_stride_lower_s = new_stride_lower;
        ratchet_.same_foot_stride_upper_s =
            (new_interval_confirmed || new_soft_upper_active)
                ? new_stride_upper
                : 0.0f;
        ratchet_.validated_stride_s = new_validated_stride;
        ratchet_.validated_riser_height_s = new_validated_riser_height;
        ratchet_.probe_bootstrap_ready = new_probe_bootstrap_ready;
        ratchet_.probe_bootstrap_provisional = new_probe_bootstrap_provisional;
        ratchet_.last_forward_up_stride = new_last_stride;
        ratchet_.stride_growth = new_growth;
        ratchet_.last_forward_up_height = new_last_height;
        ratchet_.safe_no_hit_steps = new_no_hit_steps;
        ratchet_.interval_confirmed = new_interval_confirmed;
        ratchet_.first_collision_seen = new_first_collision_seen;
        ratchet_.collision_anchor_active = new_anchor_active;
        ratchet_.collision_anchor_s = new_anchor_active ? new_anchor_s : 0.0f;
        ratchet_.collision_anchor_z = new_anchor_active ? new_anchor_z : 0.0f;
        ratchet_.collision_anchor_up_steps =
            new_anchor_active ? new_anchor_steps : 0;
        ratchet_.soft_upper_active = new_soft_upper_active;
        ratchet_.soft_upper_ttl_remaining =
            new_soft_upper_active ? new_soft_ttl : 0;
        ratchet_.mode = new_mode;
        ratchet_.upper_seen = new_upper_seen;
        ratchet_.lock_stable_count = new_lock_stable_count;
        ratchet_.lock_lower_s = new_mode == Mode::Lock ? effective_lock_lower : 0.0f;
        ratchet_.lock_upper_s = new_mode == Mode::Lock ? effective_lock_upper : 0.0f;
        ratchet_.lock_phase_error_s =
            new_mode == Mode::Lock ? new_phase_error : 0.0f;
        ratchet_.lock_phase_correction_s =
            new_mode == Mode::Lock ? lock_phase_correction : 0.0f;
        ratchet_.recovery_pending = new_recovery_pending;
        ratchet_.recovery_attempt_seen = new_recovery_attempt_seen;
        ratchet_.recovery_reduction_seen = new_recovery_reduction_seen;
        ratchet_.recovery_pending_steps = new_recovery_pending
            ? (old_recovery_pending ? ratchet_.recovery_pending_steps + 1 : 1)
            : 0;
        ratchet_.recovery_target_s =
            new_recovery_pending ? recovery_target : 0.0f;
        ratchet_.lock_target_s =
            new_interval_confirmed ? new_lock_target : 0.0f;
        const bool rejected_second_hit_pending =
            !new_interval_confirmed && rejected_second_hit_count > 0;
        ratchet_.rejected_second_hit_count =
            rejected_second_hit_pending ? rejected_second_hit_count : 0;
        ratchet_.rejected_second_hit_target_s =
            rejected_second_hit_pending ? rejected_second_hit_target : 0.0f;
        ratchet_.hold_center_active = false;
        ratchet_.hold_center_target_s = 0.0f;
        ratchet_.confidence = new_confidence;
        ratchet_.age_s = has_evidence ? 0.0f : aged;
        ratchet_.flat_pair_steps = new_flat_steps;
    }

    std::array<float, 10> ratchet_features() const
    {
        std::array<float, 10> out{};
        if (!ratchet_.active) {
            return out;
        }
        float intent = static_cast<float>(static_cast<int>(ratchet_.mode)) / 2.0f;
        const bool bootstrap_pending =
            ratchet_.first_collision_seen &&
            !ratchet_.probe_bootstrap_ready &&
            !ratchet_.interval_confirmed;
        if (bootstrap_pending) {
            intent = 1.0f;
        }
        const bool probe_active =
            ratchet_.probe_bootstrap_ready &&
            !ratchet_.interval_confirmed &&
            ratchet_.mode == Mode::Probe &&
            ratchet_.last_forward_up_stride > cfg_.ratchet_min_stride_m;
        if (probe_active) {
            const float error = ratchet_.last_forward_up_stride - ratchet_.probe_target_s;
            if (error > cfg_.ratchet_probe_target_tolerance_m) {
                intent = 0.5f;
            } else if (ratchet_.probe_target_s >= cfg_.ratchet_probe_cap_m - 1.0e-5f &&
                       std::abs(error) <= cfg_.ratchet_probe_target_tolerance_m) {
                intent = 1.0f;
            } else {
                intent = 0.0f;
            }
        }
        if (ratchet_.recovery_pending) {
            intent = 0.5f;
            if (ratchet_.recovery_attempt_seen &&
                ratchet_.last_forward_up_stride <
                    ratchet_.recovery_target_s -
                    cfg_.ratchet_recovery_completion_lower_tolerance_m) {
                intent = 0.0f;
            }
        }
        const bool lock_active =
            ratchet_.interval_confirmed &&
            !ratchet_.recovery_pending &&
            ratchet_.mode == Mode::Lock &&
            ratchet_.last_forward_up_stride > cfg_.ratchet_min_stride_m;
        if (lock_active) {
            const float error = ratchet_.last_forward_up_stride - ratchet_.lock_target_s;
            if (error > cfg_.ratchet_lock_intent_tolerance_m) {
                intent = 0.5f;
            } else if (error < -cfg_.ratchet_lock_intent_tolerance_m) {
                intent = 0.0f;
            } else {
                intent = 1.0f;
            }
        }

        out[0] = 1.0f;
        out[1] = ratchet_.lower_s;
        out[2] = ratchet_.probe_target_s;
        out[3] = ratchet_.upper_s;
        out[4] = ratchet_.last_forward_up_height;
        out[5] = ratchet_.same_foot_stride_lower_s;
        out[6] = ratchet_.interval_confirmed ? 1.0f : 0.0f;
        out[7] = ratchet_.same_foot_stride_upper_s;
        out[8] = (ratchet_.probe_bootstrap_ready || ratchet_.interval_confirmed) ? 1.0f : 0.0f;
        out[9] = intent;
        return out;
    }

    void compute_summary(
        bool update_ratchet_state,
        bool new_footprint_any,
        bool new_toe_mark_any,
        float dt)
    {
        std::fill(summary_.begin(), summary_.end(), 0.0f);
        const auto pairs = all_pair_features();
        for (int i = 0; i < kPairCount; ++i) {
            for (int j = 0; j < kPairFeatureDim; ++j) {
                summary_[static_cast<size_t>(i * kPairFeatureDim + j)] =
                    finite_or_zero(pairs[i][j]);
            }
        }
        const ToeFeatures toe = latest_toe_features();
        for (int i = 0; i < 10; ++i) {
            summary_[kToeSummaryStart + i] = finite_or_zero(toe[i]);
        }
        const auto stats = summary_stats(pairs);
        for (int i = 0; i < 10; ++i) {
            summary_[kGeometryStatsStart + i] = finite_or_zero(stats[i]);
        }

        std::array<SameFootFeatures, 6> same_steps{};
        const int same_count = std::min(6, cfg_.memory_len);
        for (int i = 0; i < same_count; ++i) {
            same_steps[i] = same_foot_step_feature(i);
        }
        if (update_ratchet_state) {
            update_ratchet(pairs, same_steps, toe, new_footprint_any, new_toe_mark_any, dt);
        }
        const auto ratchet = ratchet_features();
        for (int i = 0; i < 10; ++i) {
            summary_[kRatchetStart + i] = finite_or_zero(ratchet[i]);
        }
    }

    FootEventMemoryConfig cfg_;
    std::vector<Footprint> footprints_;
    std::vector<ToeMark> toe_marks_;
    std::vector<bool> footprint_valid_;
    std::vector<bool> toe_valid_;
    std::vector<Eigen::Vector3f> footprint_pos_w_;
    std::vector<Eigen::Vector3f> toe_pos_w_;
    std::array<bool, 2> foot_in_stance_ = {false, false};
    std::array<int, 2> release_count_ = {0, 0};
    std::array<float, 2> stance_age_s_ = {0.0f, 0.0f};
    std::array<int, 2> active_footprint_slot_ = {-1, -1};
    std::array<PredictedFillState, 2> predicted_fill_state_ = {
        PredictedFillState{},
        PredictedFillState{},
    };
    ContactFootOdometry odom_;
    RatchetState ratchet_;
    std::vector<float> summary_ = std::vector<float>(kSummaryDim, 0.0f);
};

}  // namespace mjlab::diagnostics
