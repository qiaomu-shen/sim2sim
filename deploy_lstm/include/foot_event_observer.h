// Copyright (c) 2026
// Passive ONNX foot-event observer for sim2sim diagnostics.

#pragma once

#include "onnxruntime_cxx_api.h"

#include <eigen3/Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace mjlab::diagnostics
{

struct FootEventDetectorConfig
{
    std::string name;
    std::filesystem::path model_path;
    std::string input_name = "obs_history";
    std::string output_name = "event_logits";
    std::string activation = "sigmoid";
};

struct FootEventObserverConfig
{
    bool enabled = false;
    std::filesystem::path event_path;
    std::filesystem::path frame_path;
    bool frame_log_enabled = false;
    int flush_interval_events = 10;
    int flush_interval_frames = 50;
    int feature_dim = 93;
    int history_frames = 0;
    bool history_oldest_first = true;
    FootEventDetectorConfig footprint_touchdown_detector;
    FootEventDetectorConfig toe_riser_detector;
    int left_contact_index = 0;
    int right_contact_index = 1;
    int left_touchdown_index = 2;
    int right_touchdown_index = 3;
    int left_toe_riser_index = 4;
    int right_toe_riser_index = 5;
    float touchdown_threshold = 0.5f;
    float toe_riser_threshold = 0.5f;
    float left_touchdown_threshold = -1.0f;
    float right_touchdown_threshold = -1.0f;
    float left_toe_riser_threshold = -1.0f;
    float right_toe_riser_threshold = -1.0f;
    float reset_ratio = 0.75f;
    int touchdown_confirm_frames = 1;
    int toe_riser_confirm_frames = 1;
    int touchdown_cooldown_frames = 6;
    int toe_riser_cooldown_frames = 6;
    double startup_ignore_s = 0.5;
};

struct FootEventObserverOutput
{
    float left_contact_prob = 0.0f;
    float right_contact_prob = 0.0f;
    float left_touchdown_prob = 0.0f;
    float right_touchdown_prob = 0.0f;
    float left_toe_riser_prob = 0.0f;
    float right_toe_riser_prob = 0.0f;
    bool left_touchdown_event = false;
    bool right_touchdown_event = false;
    bool left_toe_riser_event = false;
    bool right_toe_riser_event = false;
};

struct FootEventObserverFrame
{
    std::string state;
    int step = 0;
    uint32_t tick_ms = 0;
    double time_s = 0.0;
    float phase = 0.0f;
    std::array<float, 3> command = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> joystick = {0.0f, 0.0f, 0.0f};

    Eigen::Quaternionf root_quat_w = Eigen::Quaternionf::Identity();
    Eigen::Vector3f projected_gravity_b = Eigen::Vector3f::Zero();
    Eigen::Vector3f root_ang_vel_b = Eigen::Vector3f::Zero();
    Eigen::Vector3f left_toe_pos_b = Eigen::Vector3f::Zero();
    Eigen::Vector3f right_toe_pos_b = Eigen::Vector3f::Zero();
    Eigen::Vector3f left_heel_pos_b = Eigen::Vector3f::Zero();
    Eigen::Vector3f right_heel_pos_b = Eigen::Vector3f::Zero();
    std::vector<float> prev_action;
    std::vector<float> action_scale;
    Eigen::VectorXf joint_pos;
    Eigen::VectorXf default_joint_pos;
    Eigen::VectorXf joint_vel;
};

class FootEventObserver
{
public:
    void configure(const FootEventObserverConfig& config)
    {
        close();
        cfg_ = config;
        reset();
    }

    bool enabled() const
    {
        return cfg_.enabled &&
               footprint_touchdown_.session != nullptr &&
               toe_riser_.session != nullptr;
    }

    bool open()
    {
        close();
        reset();
        if (!cfg_.enabled) {
            return false;
        }

        try {
            ort_env_ = std::make_unique<Ort::Env>(
                ORT_LOGGING_LEVEL_WARNING,
                "foot_event_observer");
            session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
            footprint_touchdown_.cfg = cfg_.footprint_touchdown_detector;
            toe_riser_.cfg = cfg_.toe_riser_detector;
            open_detector(footprint_touchdown_);
            open_detector(toe_riser_);
            sync_history_contract();
        } catch (const std::exception&) {
            cfg_.enabled = false;
            close();
            return false;
        }

        if (cfg_.event_path.has_parent_path()) {
            std::filesystem::create_directories(cfg_.event_path.parent_path());
        }
        event_stream_.open(cfg_.event_path, std::ios::out | std::ios::trunc);
        if (event_stream_) {
            event_stream_.setf(std::ios::fixed);
            event_stream_ << std::setprecision(7);
            write_event_header();
            event_stream_.flush();
        }

        if (cfg_.frame_log_enabled) {
            if (cfg_.frame_path.has_parent_path()) {
                std::filesystem::create_directories(cfg_.frame_path.parent_path());
            }
            frame_stream_.open(cfg_.frame_path, std::ios::out | std::ios::trunc);
            if (frame_stream_) {
                frame_stream_.setf(std::ios::fixed);
                frame_stream_ << std::setprecision(7);
                write_frame_header();
            } else {
                cfg_.frame_log_enabled = false;
            }
        }
        return true;
    }

    void close()
    {
        if (event_stream_.is_open()) {
            event_stream_.flush();
            event_stream_.close();
        }
        if (frame_stream_.is_open()) {
            frame_stream_.flush();
            frame_stream_.close();
        }
        footprint_touchdown_ = DetectorRuntime{};
        toe_riser_ = DetectorRuntime{};
        ort_env_.reset();
    }

    void reset()
    {
        feature_history_.clear();
        event_count_ = 0;
        frame_count_ = 0;
        last_features_.clear();
        cache_valid_ = false;
        prev_toe_vel_valid_ = false;
        prev_base_ang_vel_ = Eigen::Vector3f::Zero();
        prev_left_toe_pos_b_ = Eigen::Vector3f::Zero();
        prev_right_toe_pos_b_ = Eigen::Vector3f::Zero();
        prev_left_toe_vel_b_ = Eigen::Vector3f::Zero();
        prev_right_toe_vel_b_ = Eigen::Vector3f::Zero();
        prev_leg_joint_vel_.clear();
        left_touchdown_ = EventState{};
        right_touchdown_ = EventState{};
        left_toe_riser_ = EventState{};
        right_toe_riser_ = EventState{};
        latest_output_ = FootEventObserverOutput{};
    }

    FootEventObserverOutput update(const FootEventObserverFrame& frame)
    {
        FootEventObserverOutput output;
        if (!enabled()) {
            return output;
        }

        const std::vector<float> features = make_stair_latent_features(frame);
        last_features_ = features;
        feature_history_.push_back(features);
        while (static_cast<int>(feature_history_.size()) > history_frames_) {
            feature_history_.pop_front();
        }

        std::vector<float> input_data = make_input_tensor(features);
        sanitize(input_data);

        std::vector<float> footprint_probabilities;
        std::vector<float> toe_probabilities;
        try {
            footprint_probabilities = run_model(footprint_touchdown_, input_data);
            toe_probabilities = run_model(toe_riser_, input_data);
        } catch (const std::exception& e) {
            cfg_.enabled = false;
            close();
            throw std::runtime_error(
                "Foot event observer inference failed: " + std::string(e.what()));
        }

        const float left_contact =
            probability_at(footprint_probabilities, cfg_.left_contact_index);
        const float right_contact =
            probability_at(footprint_probabilities, cfg_.right_contact_index);
        const float left_td =
            probability_at(footprint_probabilities, cfg_.left_touchdown_index);
        const float right_td =
            probability_at(footprint_probabilities, cfg_.right_touchdown_index);
        const float left_toe =
            probability_at(toe_probabilities, cfg_.left_toe_riser_index);
        const float right_toe =
            probability_at(toe_probabilities, cfg_.right_toe_riser_index);
        output.left_contact_prob = left_contact;
        output.right_contact_prob = right_contact;
        output.left_touchdown_prob = left_td;
        output.right_touchdown_prob = right_td;
        output.left_toe_riser_prob = left_toe;
        output.right_toe_riser_prob = right_toe;
        latest_output_ = output;

        if (cfg_.frame_log_enabled && frame_stream_.is_open()) {
            write_frame(frame, left_td, right_td, left_toe, right_toe);
        }

        if (frame.time_s < cfg_.startup_ignore_s) {
            latest_output_ = output;
            return output;
        }

        output.left_touchdown_event = update_event(
            "observer_touchdown",
            "left",
            left_td,
            threshold_or_default(
                cfg_.left_touchdown_threshold,
                cfg_.touchdown_threshold),
            cfg_.touchdown_confirm_frames,
            cfg_.touchdown_cooldown_frames,
            cfg_.left_touchdown_index,
            frame,
            left_touchdown_,
            left_td,
            right_td,
            left_toe,
            right_toe);
        output.right_touchdown_event = update_event(
            "observer_touchdown",
            "right",
            right_td,
            threshold_or_default(
                cfg_.right_touchdown_threshold,
                cfg_.touchdown_threshold),
            cfg_.touchdown_confirm_frames,
            cfg_.touchdown_cooldown_frames,
            cfg_.right_touchdown_index,
            frame,
            right_touchdown_,
            left_td,
            right_td,
            left_toe,
            right_toe);
        output.left_toe_riser_event = update_event(
            "observer_toe_riser",
            "left",
            left_toe,
            threshold_or_default(
                cfg_.left_toe_riser_threshold,
                cfg_.toe_riser_threshold),
            cfg_.toe_riser_confirm_frames,
            cfg_.toe_riser_cooldown_frames,
            cfg_.left_toe_riser_index,
            frame,
            left_toe_riser_,
            left_td,
            right_td,
            left_toe,
            right_toe);
        output.right_toe_riser_event = update_event(
            "observer_toe_riser",
            "right",
            right_toe,
            threshold_or_default(
                cfg_.right_toe_riser_threshold,
                cfg_.toe_riser_threshold),
            cfg_.toe_riser_confirm_frames,
            cfg_.toe_riser_cooldown_frames,
            cfg_.right_toe_riser_index,
            frame,
            right_toe_riser_,
            left_td,
            right_td,
            left_toe,
            right_toe);
        latest_output_ = output;
        return output;
    }

    FootEventObserverOutput latest_output() const
    {
        return latest_output_;
    }

#ifdef MJLAB_FOOT_EVENT_OBSERVER_TESTING
    size_t history_size_for_testing() const
    {
        return feature_history_.size();
    }

    const std::vector<float>& last_features_for_testing() const
    {
        return last_features_;
    }
#endif

private:
    struct EventState
    {
        int high_frames = 0;
        bool armed = true;
        int last_event_step = -1000000;
    };

    struct DetectorRuntime
    {
        FootEventDetectorConfig cfg;
        std::unique_ptr<Ort::Session> session;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::vector<const char*> input_name_ptrs;
        std::vector<const char*> output_name_ptrs;
        std::vector<std::vector<int64_t>> input_shapes;
        std::vector<std::vector<int64_t>> output_shapes;
        std::vector<size_t> input_sizes;
        std::vector<size_t> output_sizes;
        std::vector<std::vector<float>> state_inputs;
        int input_index = 0;
        int output_index = 0;
        size_t input_size = 0;
        size_t output_size = 0;
    };

    static size_t tensor_size(const std::vector<int64_t>& shape)
    {
        size_t size = 1;
        for (const int64_t dim : shape) {
            if (dim <= 0) {
                throw std::runtime_error(
                    "Only the batch dimension may be dynamic in foot event observer ONNX tensors.");
            }
            size *= static_cast<size_t>(dim);
        }
        return size;
    }

    static std::vector<int64_t> normalize_shape(std::vector<int64_t> shape)
    {
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] <= 0 && i == 0) {
                shape[i] = 1;
            }
        }
        return shape;
    }

    static int find_name(const std::vector<std::string>& names, const std::string& name)
    {
        if (name.empty()) {
            return -1;
        }
        const auto it = std::find(names.begin(), names.end(), name);
        if (it == names.end()) {
            return -1;
        }
        return static_cast<int>(std::distance(names.begin(), it));
    }

    static bool has_suffix(const std::string& value, const std::string& suffix)
    {
        return value.size() >= suffix.size() &&
               value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    static bool looks_like_state_input(const std::string& name)
    {
        return name == "h_in" ||
               name == "c_in" ||
               name == "z_in" ||
               name == "gate_state_in" ||
               has_suffix(name, "_state_in");
    }

    static bool looks_like_state_output(const std::string& name)
    {
        return name == "h_out" ||
               name == "c_out" ||
               name == "z_out" ||
               name == "gate_state_out" ||
               has_suffix(name, "_state_out");
    }

    static int first_non_state_name(
        const std::vector<std::string>& names,
        bool (*is_state_name)(const std::string&))
    {
        for (size_t i = 0; i < names.size(); ++i) {
            if (!is_state_name(names[i])) {
                return static_cast<int>(i);
            }
        }
        return names.empty() ? -1 : 0;
    }

    static void append_vec3(std::ofstream& stream, const Eigen::Vector3f& value)
    {
        stream << ',' << value.x()
               << ',' << value.y()
               << ',' << value.z();
    }

    static float sigmoid(float value)
    {
        if (value >= 0.0f) {
            const float z = std::exp(-value);
            return 1.0f / (1.0f + z);
        }
        const float z = std::exp(value);
        return z / (1.0f + z);
    }

    void open_detector(DetectorRuntime& detector)
    {
        if (detector.cfg.name.empty()) {
            throw std::runtime_error("Foot event detector name is empty.");
        }
        if (!std::filesystem::exists(detector.cfg.model_path)) {
            throw std::runtime_error(
                "Foot event detector model not found: " +
                detector.cfg.model_path.string());
        }
        detector.session = std::make_unique<Ort::Session>(
            *ort_env_,
            detector.cfg.model_path.c_str(),
            session_options_);
        inspect_detector(detector);
    }

    void inspect_detector(DetectorRuntime& detector)
    {
        detector.input_names.clear();
        detector.output_names.clear();
        detector.input_shapes.clear();
        detector.output_shapes.clear();
        detector.input_sizes.clear();
        detector.output_sizes.clear();

        for (size_t i = 0; i < detector.session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = detector.session->GetInputTypeInfo(i);
            auto shape = normalize_shape(
                input_type.GetTensorTypeAndShapeInfo().GetShape());
            detector.input_shapes.push_back(shape);
            detector.input_sizes.push_back(tensor_size(shape));
            auto input_name = detector.session->GetInputNameAllocated(i, allocator_);
            detector.input_names.push_back(input_name.get());
        }

        for (size_t i = 0; i < detector.session->GetOutputCount(); ++i) {
            Ort::TypeInfo output_type = detector.session->GetOutputTypeInfo(i);
            auto shape = normalize_shape(
                output_type.GetTensorTypeAndShapeInfo().GetShape());
            detector.output_shapes.push_back(shape);
            detector.output_sizes.push_back(tensor_size(shape));
            auto output_name = detector.session->GetOutputNameAllocated(i, allocator_);
            detector.output_names.push_back(output_name.get());
        }

        detector.input_index = find_name(detector.input_names, detector.cfg.input_name);
        if (detector.input_index < 0) {
            detector.input_index =
                first_non_state_name(detector.input_names, looks_like_state_input);
        }
        detector.output_index = find_name(detector.output_names, detector.cfg.output_name);
        if (detector.output_index < 0) {
            detector.output_index =
                first_non_state_name(detector.output_names, looks_like_state_output);
        }

        detector.input_name_ptrs.clear();
        detector.input_name_ptrs.reserve(detector.input_names.size());
        for (const auto& name : detector.input_names) {
            detector.input_name_ptrs.push_back(name.c_str());
        }
        detector.output_name_ptrs.clear();
        detector.output_name_ptrs.reserve(detector.output_names.size());
        for (const auto& name : detector.output_names) {
            detector.output_name_ptrs.push_back(name.c_str());
        }

        detector.input_size = detector.input_sizes.at(detector.input_index);
        detector.output_size = detector.output_sizes.at(detector.output_index);
        if (detector.input_size == 0 || detector.output_size == 0) {
            throw std::runtime_error(
                "Empty foot event observer ONNX tensor in " + detector.cfg.name);
        }

        detector.state_inputs.clear();
        detector.state_inputs.resize(detector.input_names.size());
        for (size_t i = 0; i < detector.input_names.size(); ++i) {
            if (static_cast<int>(i) == detector.input_index) {
                continue;
            }
            detector.state_inputs[i].resize(detector.input_sizes[i], 0.0f);
        }
    }

    void sync_history_contract()
    {
        const int feature_dim = std::max(1, cfg_.feature_dim);
        const auto derive_history = [feature_dim](const DetectorRuntime& detector) {
            if (detector.input_size % static_cast<size_t>(feature_dim) != 0) {
                throw std::runtime_error(
                    "Detector input size is not divisible by feature_dim: " +
                    detector.cfg.name);
            }
            return static_cast<int>(detector.input_size / static_cast<size_t>(feature_dim));
        };
        const int footprint_history = derive_history(footprint_touchdown_);
        const int toe_history = derive_history(toe_riser_);
        if (footprint_history != toe_history) {
            throw std::runtime_error("Foot event detector history lengths differ.");
        }
        history_frames_ = cfg_.history_frames > 0
            ? cfg_.history_frames
            : footprint_history;
        if (history_frames_ != footprint_history || history_frames_ != toe_history) {
            throw std::runtime_error(
                "Configured foot event history length does not match detector ONNX input.");
        }
        history_frames_ = std::max(1, history_frames_);
    }

    std::vector<float> make_stair_latent_features(
        const FootEventObserverFrame& frame)
    {
        std::vector<float> out;
        out.reserve(static_cast<size_t>(std::max(91, cfg_.feature_dim)));

        auto push3 = [&out](const Eigen::Vector3f& value) {
            out.push_back(value.x());
            out.push_back(value.y());
            out.push_back(value.z());
        };

        const Eigen::Vector3f toe_delta =
            frame.left_toe_pos_b - frame.right_toe_pos_b;
        const Eigen::Vector3f heel_delta =
            frame.left_heel_pos_b - frame.right_heel_pos_b;
        const Eigen::Vector3f toe_delta_w = frame.root_quat_w * toe_delta;
        const Eigen::Vector3f heel_delta_w = frame.root_quat_w * heel_delta;

        Eigen::Vector3f base_ang_vel_delta = Eigen::Vector3f::Zero();
        Eigen::Vector3f left_toe_vel = Eigen::Vector3f::Zero();
        Eigen::Vector3f right_toe_vel = Eigen::Vector3f::Zero();
        if (cache_valid_) {
            base_ang_vel_delta = frame.root_ang_vel_b - prev_base_ang_vel_;
            left_toe_vel =
                (frame.left_toe_pos_b - prev_left_toe_pos_b_) / frame_dt(frame);
            right_toe_vel =
                (frame.right_toe_pos_b - prev_right_toe_pos_b_) / frame_dt(frame);
        }

        Eigen::Vector3f left_toe_vel_delta = Eigen::Vector3f::Zero();
        Eigen::Vector3f right_toe_vel_delta = Eigen::Vector3f::Zero();
        if (cache_valid_ && prev_toe_vel_valid_) {
            left_toe_vel_delta = left_toe_vel - prev_left_toe_vel_b_;
            right_toe_vel_delta = right_toe_vel - prev_right_toe_vel_b_;
        }

        std::vector<float> prev_action_leg(12, 0.0f);
        for (int i = 0; i < 12 && i < static_cast<int>(frame.prev_action.size()); ++i) {
            prev_action_leg[i] = frame.prev_action[i];
        }

        std::vector<float> leg_tracking_error(12, 0.0f);
        for (int i = 0; i < 12; ++i) {
            const float measured =
                i < frame.joint_pos.size() ? frame.joint_pos[i] : 0.0f;
            const float q_default =
                i < frame.default_joint_pos.size() ? frame.default_joint_pos[i] : 0.0f;
            const float scale =
                i < static_cast<int>(frame.action_scale.size()) ? frame.action_scale[i] : 1.0f;
            leg_tracking_error[i] = q_default + prev_action_leg[i] * scale - measured;
        }

        std::vector<float> leg_joint_vel(12, 0.0f);
        for (int i = 0; i < 12; ++i) {
            leg_joint_vel[i] = i < frame.joint_vel.size() ? frame.joint_vel[i] : 0.0f;
        }

        std::vector<float> leg_joint_vel_delta(12, 0.0f);
        if (cache_valid_ && prev_leg_joint_vel_.size() == 12) {
            for (int i = 0; i < 12; ++i) {
                leg_joint_vel_delta[i] = leg_joint_vel[i] - prev_leg_joint_vel_[i];
            }
        }

        push3(frame.projected_gravity_b);
        push3(frame.root_ang_vel_b);
        push3(base_ang_vel_delta);
        push3(frame.left_toe_pos_b);
        push3(frame.right_toe_pos_b);
        push3(frame.left_heel_pos_b);
        push3(frame.right_heel_pos_b);
        push3(toe_delta);
        push3(heel_delta);
        out.push_back(std::abs(toe_delta.x()));
        out.push_back(toe_delta_w.z());
        out.push_back(heel_delta_w.z());
        push3(left_toe_vel);
        push3(right_toe_vel);
        push3(left_toe_vel_delta);
        push3(right_toe_vel_delta);
        out.insert(out.end(), prev_action_leg.begin(), prev_action_leg.end());
        out.insert(out.end(), leg_tracking_error.begin(), leg_tracking_error.end());
        out.insert(out.end(), leg_joint_vel.begin(), leg_joint_vel.end());
        out.insert(out.end(), leg_joint_vel_delta.begin(), leg_joint_vel_delta.end());
        out.push_back(frame.command[0]);
        const float command_norm = std::sqrt(
            frame.command[0] * frame.command[0] +
            frame.command[1] * frame.command[1] +
            frame.command[2] * frame.command[2]);
        if (command_norm < 0.1f) {
            out.push_back(0.0f);
            out.push_back(0.0f);
        } else {
            out.push_back(std::sin(frame.phase * 2.0f * static_cast<float>(M_PI)));
            out.push_back(std::cos(frame.phase * 2.0f * static_cast<float>(M_PI)));
        }

        prev_base_ang_vel_ = frame.root_ang_vel_b;
        prev_left_toe_pos_b_ = frame.left_toe_pos_b;
        prev_right_toe_pos_b_ = frame.right_toe_pos_b;
        prev_left_toe_vel_b_ = left_toe_vel;
        prev_right_toe_vel_b_ = right_toe_vel;
        prev_leg_joint_vel_ = leg_joint_vel;
        prev_toe_vel_valid_ = true;
        cache_valid_ = true;
        last_step_ = frame.step;
        last_time_s_ = frame.time_s;

        const int feature_dim = std::max(1, cfg_.feature_dim);
        if (static_cast<int>(out.size()) < feature_dim) {
            out.resize(feature_dim, 0.0f);
        } else if (static_cast<int>(out.size()) > feature_dim) {
            out.resize(feature_dim);
        }
        return out;
    }

    float frame_dt(const FootEventObserverFrame& frame) const
    {
        if (last_step_ >= 0 && frame.step > last_step_) {
            const float inferred = static_cast<float>(frame.time_s - last_time_s_);
            if (std::isfinite(inferred) && inferred > 1.0e-6f) {
                return inferred;
            }
        }
        return 0.02f;
    }

    std::vector<float> make_input_tensor(const std::vector<float>& current_features)
    {
        std::vector<float> input;
        const size_t expected_size =
            static_cast<size_t>(history_frames_) *
            static_cast<size_t>(std::max(1, cfg_.feature_dim));
        input.reserve(expected_size);

        std::deque<std::vector<float>> padded = feature_history_;
        while (static_cast<int>(padded.size()) < history_frames_) {
            padded.push_front(std::vector<float>(current_features.size(), 0.0f));
        }

        if (cfg_.history_oldest_first) {
            for (const auto& features : padded) {
                input.insert(input.end(), features.begin(), features.end());
            }
        } else {
            for (auto it = padded.rbegin(); it != padded.rend(); ++it) {
                input.insert(input.end(), it->begin(), it->end());
            }
        }

        if (input.size() < expected_size) {
            input.resize(expected_size, 0.0f);
        } else if (input.size() > expected_size) {
            input.resize(expected_size);
        }
        return input;
    }

    static void sanitize(std::vector<float>& values)
    {
        for (float& value : values) {
            if (!std::isfinite(value)) {
                value = 0.0f;
            }
        }
    }

    std::vector<float> run_model(DetectorRuntime& detector, std::vector<float>& input_data)
    {
        auto memory_info =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(detector.input_names.size());
        for (size_t i = 0; i < detector.input_names.size(); ++i) {
            if (static_cast<int>(i) == detector.input_index) {
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    memory_info,
                    input_data.data(),
                    input_data.size(),
                    detector.input_shapes[i].data(),
                    detector.input_shapes[i].size()));
            } else {
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    memory_info,
                    detector.state_inputs[i].data(),
                    detector.state_inputs[i].size(),
                    detector.input_shapes[i].data(),
                    detector.input_shapes[i].size()));
            }
        }

        auto output_tensors = detector.session->Run(
            Ort::RunOptions{nullptr},
            detector.input_name_ptrs.data(),
            input_tensors.data(),
            input_tensors.size(),
            detector.output_name_ptrs.data(),
            detector.output_name_ptrs.size());

        auto* output_data =
            output_tensors[detector.output_index].GetTensorMutableData<float>();
        std::vector<float> values(output_data, output_data + detector.output_size);
        update_state_inputs(detector, output_tensors);
        apply_activation(detector, values);
        return values;
    }

    void update_state_inputs(
        DetectorRuntime& detector,
        std::vector<Ort::Value>& output_tensors)
    {
        for (size_t i = 0; i < detector.input_names.size(); ++i) {
            if (static_cast<int>(i) == detector.input_index ||
                detector.state_inputs[i].empty()) {
                continue;
            }
            const int out_idx = matching_state_output(detector, detector.input_names[i]);
            if (out_idx < 0 ||
                detector.output_sizes[out_idx] != detector.state_inputs[i].size()) {
                continue;
            }
            auto* out = output_tensors[out_idx].GetTensorMutableData<float>();
            std::copy(
                out,
                out + detector.state_inputs[i].size(),
                detector.state_inputs[i].begin());
        }
    }

    int matching_state_output(
        const DetectorRuntime& detector,
        const std::string& input_name) const
    {
        if (input_name.size() > 3 &&
            input_name.substr(input_name.size() - 3) == "_in") {
            std::string output_name = input_name;
            output_name.replace(output_name.size() - 3, 3, "_out");
            const int idx = find_name(detector.output_names, output_name);
            if (idx >= 0) {
                return idx;
            }
        }
        const int exact = find_name(detector.output_names, input_name);
        return exact;
    }

    void apply_activation(
        const DetectorRuntime& detector,
        std::vector<float>& values) const
    {
        if (detector.cfg.activation == "none" ||
            detector.cfg.activation == "identity") {
            return;
        }
        if (detector.cfg.activation == "softmax") {
            const float max_value =
                *std::max_element(values.begin(), values.end());
            float denom = 0.0f;
            for (float& value : values) {
                value = std::exp(value - max_value);
                denom += value;
            }
            if (denom > 0.0f) {
                for (float& value : values) {
                    value /= denom;
                }
            }
            return;
        }
        if (detector.cfg.activation == "auto") {
            const bool already_probabilities = std::all_of(
                values.begin(),
                values.end(),
                [](float value) {
                    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
                });
            if (already_probabilities) {
                return;
            }
        }
        for (float& value : values) {
            value = sigmoid(value);
        }
    }

    static float probability_at(const std::vector<float>& values, int index)
    {
        if (index < 0 || index >= static_cast<int>(values.size())) {
            return 0.0f;
        }
        return values[index];
    }

    static float threshold_or_default(float value, float fallback)
    {
        return value >= 0.0f ? value : fallback;
    }

    bool update_event(
        const char* event_type,
        const char* foot,
        float probability,
        float threshold,
        int confirm_frames,
        int cooldown_frames,
        int output_index,
        const FootEventObserverFrame& frame,
        EventState& state,
        float left_td,
        float right_td,
        float left_toe,
        float right_toe)
    {
        const float reset_threshold = threshold * cfg_.reset_ratio;
        if (probability < reset_threshold) {
            state.armed = true;
        }

        if (probability >= threshold) {
            state.high_frames += 1;
        } else {
            state.high_frames = 0;
            return false;
        }

        const bool confirmed = state.high_frames >= std::max(1, confirm_frames);
        const bool cooled_down =
            frame.step - state.last_event_step >= std::max(1, cooldown_frames);
        if (!state.armed || !confirmed || !cooled_down) {
            return false;
        }

        write_event(
            event_type,
            foot,
            probability,
            threshold,
            output_index,
            frame,
            left_td,
            right_td,
            left_toe,
            right_toe);
        state.armed = false;
        state.last_event_step = frame.step;
        return true;
    }

    void write_event_header()
    {
        if (!event_stream_.is_open()) {
            return;
        }
        event_stream_
            << "event_type,foot,event_step,event_tick_ms,event_time_s"
            << ",state,phase,probability,threshold,output_index"
            << ",cmd_x,cmd_y,cmd_yaw,stick_ly,stick_lx,stick_rx"
            << ",foot_pos_b_x,foot_pos_b_y,foot_pos_b_z"
            << ",toe_pos_b_x,toe_pos_b_y,toe_pos_b_z"
            << ",heel_pos_b_x,heel_pos_b_y,heel_pos_b_z"
            << ",left_contact_prob,right_contact_prob"
            << ",left_touchdown_prob,right_touchdown_prob"
            << ",left_toe_riser_prob,right_toe_riser_prob"
            << ",footprint_touchdown_model_path,toe_riser_model_path"
            << ",footprint_output_name,toe_output_name\n";
    }

    void write_frame_header()
    {
        if (!frame_stream_.is_open()) {
            return;
        }
        frame_stream_
            << "state,step,time_s,lowstate_tick_ms,phase"
            << ",cmd_x,cmd_y,cmd_yaw,stick_ly,stick_lx,stick_rx"
            << ",left_contact_prob,right_contact_prob"
            << ",left_touchdown_prob,right_touchdown_prob"
            << ",left_toe_riser_prob,right_toe_riser_prob\n";
    }

    void write_event(
        const char* event_type,
        const char* foot,
        float probability,
        float threshold,
        int output_index,
        const FootEventObserverFrame& frame,
        float left_td,
        float right_td,
        float left_toe,
        float right_toe)
    {
        if (!event_stream_.is_open()) {
            return;
        }
        const bool left = std::string(foot) == "left";
        const Eigen::Vector3f toe =
            left ? frame.left_toe_pos_b : frame.right_toe_pos_b;
        const Eigen::Vector3f heel =
            left ? frame.left_heel_pos_b : frame.right_heel_pos_b;
        const Eigen::Vector3f foot_pos = 0.5f * (toe + heel);

        event_stream_
            << event_type
            << ',' << foot
            << ',' << frame.step
            << ',' << frame.tick_ms
            << ',' << frame.time_s
            << ',' << frame.state
            << ',' << frame.phase
            << ',' << probability
            << ',' << threshold
            << ',' << output_index
            << ',' << frame.command[0]
            << ',' << frame.command[1]
            << ',' << frame.command[2]
            << ',' << frame.joystick[0]
            << ',' << frame.joystick[1]
            << ',' << frame.joystick[2];
        append_vec3(event_stream_, foot_pos);
        append_vec3(event_stream_, toe);
        append_vec3(event_stream_, heel);
        event_stream_
            << ',' << latest_output_.left_contact_prob
            << ',' << latest_output_.right_contact_prob
            << ',' << left_td
            << ',' << right_td
            << ',' << left_toe
            << ',' << right_toe
            << ',' << footprint_touchdown_.cfg.model_path.string()
            << ',' << toe_riser_.cfg.model_path.string()
            << ',' << footprint_touchdown_.output_names.at(footprint_touchdown_.output_index)
            << ',' << toe_riser_.output_names.at(toe_riser_.output_index)
            << '\n';

        event_count_ += 1;
        if (event_count_ % std::max(1, cfg_.flush_interval_events) == 0) {
            event_stream_.flush();
        }
    }

    void write_frame(
        const FootEventObserverFrame& frame,
        float left_td,
        float right_td,
        float left_toe,
        float right_toe)
    {
        if (!frame_stream_.is_open()) {
            return;
        }
        frame_stream_
            << frame.state
            << ',' << frame.step
            << ',' << frame.time_s
            << ',' << frame.tick_ms
            << ',' << frame.phase
            << ',' << frame.command[0]
            << ',' << frame.command[1]
            << ',' << frame.command[2]
            << ',' << frame.joystick[0]
            << ',' << frame.joystick[1]
            << ',' << frame.joystick[2]
            << ',' << latest_output_.left_contact_prob
            << ',' << latest_output_.right_contact_prob
            << ',' << left_td
            << ',' << right_td
            << ',' << left_toe
            << ',' << right_toe
            << '\n';
        frame_count_ += 1;
        if (frame_count_ % std::max(1, cfg_.flush_interval_frames) == 0) {
            frame_stream_.flush();
        }
    }

    FootEventObserverConfig cfg_;
    std::unique_ptr<Ort::Env> ort_env_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;
    DetectorRuntime footprint_touchdown_;
    DetectorRuntime toe_riser_;
    int history_frames_ = 1;

    std::ofstream event_stream_;
    std::ofstream frame_stream_;
    int event_count_ = 0;
    int frame_count_ = 0;

    std::deque<std::vector<float>> feature_history_;
    std::vector<float> last_features_;
    bool cache_valid_ = false;
    bool prev_toe_vel_valid_ = false;
    int last_step_ = -1;
    double last_time_s_ = 0.0;
    Eigen::Vector3f prev_base_ang_vel_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_left_toe_pos_b_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_right_toe_pos_b_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_left_toe_vel_b_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_right_toe_vel_b_ = Eigen::Vector3f::Zero();
    std::vector<float> prev_leg_joint_vel_;

    EventState left_touchdown_;
    EventState right_touchdown_;
    EventState left_toe_riser_;
    EventState right_toe_riser_;
    FootEventObserverOutput latest_output_;
};

}  // namespace mjlab::diagnostics
