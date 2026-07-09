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

struct FootEventObserverConfig
{
    bool enabled = false;
    std::filesystem::path model_path;
    std::filesystem::path event_path;
    std::filesystem::path frame_path;
    bool frame_log_enabled = false;
    int flush_interval_events = 10;
    int flush_interval_frames = 50;
    int feature_dim = 93;
    int history_frames = 0;
    bool history_oldest_first = true;
    std::string input_name;
    std::string output_name;
    std::string activation = "sigmoid";
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
        return cfg_.enabled && session_ != nullptr && event_stream_.is_open();
    }

    bool open()
    {
        close();
        reset();
        if (!cfg_.enabled) {
            return false;
        }
        if (!std::filesystem::exists(cfg_.model_path)) {
            cfg_.enabled = false;
            return false;
        }

        try {
            ort_env_ = std::make_unique<Ort::Env>(
                ORT_LOGGING_LEVEL_WARNING,
                "foot_event_observer");
            session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
            session_ = std::make_unique<Ort::Session>(
                *ort_env_,
                cfg_.model_path.c_str(),
                session_options_);
            inspect_model();
        } catch (const std::exception&) {
            cfg_.enabled = false;
            session_.reset();
            ort_env_.reset();
            return false;
        }

        if (cfg_.event_path.has_parent_path()) {
            std::filesystem::create_directories(cfg_.event_path.parent_path());
        }
        event_stream_.open(cfg_.event_path, std::ios::out | std::ios::trunc);
        if (!event_stream_) {
            cfg_.enabled = false;
            session_.reset();
            ort_env_.reset();
            return false;
        }
        event_stream_.setf(std::ios::fixed);
        event_stream_ << std::setprecision(7);
        write_event_header();
        event_stream_.flush();

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
        session_.reset();
        ort_env_.reset();
        input_name_ptrs_.clear();
        output_name_ptrs_.clear();
        input_names_.clear();
        output_names_.clear();
        input_shapes_.clear();
        output_shapes_.clear();
        input_sizes_.clear();
        output_sizes_.clear();
    }

    void reset()
    {
        feature_history_.clear();
        event_count_ = 0;
        frame_count_ = 0;
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
    }

    void update(const FootEventObserverFrame& frame)
    {
        if (!enabled()) {
            return;
        }

        const std::vector<float> features = make_stair_latent_features(frame);
        feature_history_.push_back(features);
        while (static_cast<int>(feature_history_.size()) > history_frames_) {
            feature_history_.pop_front();
        }

        std::vector<float> input_data = make_input_tensor(features);
        sanitize(input_data);

        std::vector<float> probabilities;
        try {
            probabilities = run_model(input_data);
        } catch (const std::exception&) {
            cfg_.enabled = false;
            return;
        }

        const float left_td = probability_at(probabilities, cfg_.left_touchdown_index);
        const float right_td = probability_at(probabilities, cfg_.right_touchdown_index);
        const float left_toe = probability_at(probabilities, cfg_.left_toe_riser_index);
        const float right_toe = probability_at(probabilities, cfg_.right_toe_riser_index);

        if (cfg_.frame_log_enabled && frame_stream_.is_open()) {
            write_frame(frame, left_td, right_td, left_toe, right_toe);
        }

        if (frame.time_s < cfg_.startup_ignore_s) {
            return;
        }

        update_event(
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
        update_event(
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
        update_event(
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
        update_event(
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
    }

private:
    struct EventState
    {
        int high_frames = 0;
        bool armed = true;
        int last_event_step = -1000000;
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

    void inspect_model()
    {
        input_names_.clear();
        output_names_.clear();
        input_shapes_.clear();
        output_shapes_.clear();
        input_sizes_.clear();
        output_sizes_.clear();

        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session_->GetInputTypeInfo(i);
            auto shape = normalize_shape(
                input_type.GetTensorTypeAndShapeInfo().GetShape());
            input_shapes_.push_back(shape);
            input_sizes_.push_back(tensor_size(shape));
            auto input_name = session_->GetInputNameAllocated(i, allocator_);
            input_names_.push_back(input_name.get());
        }

        for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
            Ort::TypeInfo output_type = session_->GetOutputTypeInfo(i);
            auto shape = normalize_shape(
                output_type.GetTensorTypeAndShapeInfo().GetShape());
            output_shapes_.push_back(shape);
            output_sizes_.push_back(tensor_size(shape));
            auto output_name = session_->GetOutputNameAllocated(i, allocator_);
            output_names_.push_back(output_name.get());
        }

        input_index_ = find_name(input_names_, cfg_.input_name);
        if (input_index_ < 0) {
            input_index_ = first_non_state_name(input_names_, looks_like_state_input);
        }
        output_index_ = find_name(output_names_, cfg_.output_name);
        if (output_index_ < 0) {
            output_index_ = first_non_state_name(output_names_, looks_like_state_output);
        }

        input_name_ptrs_.clear();
        input_name_ptrs_.reserve(input_names_.size());
        for (const auto& name : input_names_) {
            input_name_ptrs_.push_back(name.c_str());
        }
        output_name_ptrs_.clear();
        output_name_ptrs_.reserve(output_names_.size());
        for (const auto& name : output_names_) {
            output_name_ptrs_.push_back(name.c_str());
        }

        input_size_ = input_sizes_.at(input_index_);
        output_size_ = output_sizes_.at(output_index_);
        if (input_size_ == 0 || output_size_ == 0) {
            throw std::runtime_error("Empty foot event observer ONNX tensor.");
        }

        state_inputs_.clear();
        state_inputs_.resize(input_names_.size());
        for (size_t i = 0; i < input_names_.size(); ++i) {
            if (static_cast<int>(i) == input_index_) {
                continue;
            }
            state_inputs_[i].resize(input_sizes_[i], 0.0f);
        }

        const int feature_dim = std::max(1, cfg_.feature_dim);
        if (cfg_.history_frames > 0) {
            history_frames_ = cfg_.history_frames;
        } else if (input_size_ % static_cast<size_t>(feature_dim) == 0) {
            history_frames_ = static_cast<int>(input_size_ / feature_dim);
        } else {
            history_frames_ = 1;
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
        input.reserve(input_size_);

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

        if (input.size() < input_size_) {
            input.resize(input_size_, 0.0f);
        } else if (input.size() > input_size_) {
            input.resize(input_size_);
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

    std::vector<float> run_model(std::vector<float>& input_data)
    {
        auto memory_info =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(input_names_.size());
        for (size_t i = 0; i < input_names_.size(); ++i) {
            if (static_cast<int>(i) == input_index_) {
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    memory_info,
                    input_data.data(),
                    input_data.size(),
                    input_shapes_[i].data(),
                    input_shapes_[i].size()));
            } else {
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    memory_info,
                    state_inputs_[i].data(),
                    state_inputs_[i].size(),
                    input_shapes_[i].data(),
                    input_shapes_[i].size()));
            }
        }

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_name_ptrs_.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_name_ptrs_.data(),
            output_name_ptrs_.size());

        auto* output_data =
            output_tensors[output_index_].GetTensorMutableData<float>();
        std::vector<float> values(output_data, output_data + output_size_);
        update_state_inputs(output_tensors);
        apply_activation(values);
        return values;
    }

    void update_state_inputs(std::vector<Ort::Value>& output_tensors)
    {
        for (size_t i = 0; i < input_names_.size(); ++i) {
            if (static_cast<int>(i) == input_index_ || state_inputs_[i].empty()) {
                continue;
            }
            const int out_idx = matching_state_output(input_names_[i]);
            if (out_idx < 0 ||
                output_sizes_[out_idx] != state_inputs_[i].size()) {
                continue;
            }
            auto* out = output_tensors[out_idx].GetTensorMutableData<float>();
            std::copy(out, out + state_inputs_[i].size(), state_inputs_[i].begin());
        }
    }

    int matching_state_output(const std::string& input_name) const
    {
        if (input_name.size() > 3 &&
            input_name.substr(input_name.size() - 3) == "_in") {
            std::string output_name = input_name;
            output_name.replace(output_name.size() - 3, 3, "_out");
            const int idx = find_name(output_names_, output_name);
            if (idx >= 0) {
                return idx;
            }
        }
        const int exact = find_name(output_names_, input_name);
        return exact;
    }

    void apply_activation(std::vector<float>& values) const
    {
        if (cfg_.activation == "none" || cfg_.activation == "identity") {
            return;
        }
        if (cfg_.activation == "softmax") {
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
        if (cfg_.activation == "auto") {
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

    void update_event(
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
            return;
        }

        const bool confirmed = state.high_frames >= std::max(1, confirm_frames);
        const bool cooled_down =
            frame.step - state.last_event_step >= std::max(1, cooldown_frames);
        if (!state.armed || !confirmed || !cooled_down) {
            return;
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
    }

    void write_event_header()
    {
        event_stream_
            << "event_type,foot,event_step,event_tick_ms,event_time_s"
            << ",state,phase,probability,threshold,output_index"
            << ",cmd_x,cmd_y,cmd_yaw,stick_ly,stick_lx,stick_rx"
            << ",foot_pos_b_x,foot_pos_b_y,foot_pos_b_z"
            << ",toe_pos_b_x,toe_pos_b_y,toe_pos_b_z"
            << ",heel_pos_b_x,heel_pos_b_y,heel_pos_b_z"
            << ",left_touchdown_prob,right_touchdown_prob"
            << ",left_toe_riser_prob,right_toe_riser_prob"
            << ",model_path,input_name,output_name\n";
    }

    void write_frame_header()
    {
        frame_stream_
            << "state,step,time_s,lowstate_tick_ms,phase"
            << ",cmd_x,cmd_y,cmd_yaw,stick_ly,stick_lx,stick_rx"
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
            << ',' << left_td
            << ',' << right_td
            << ',' << left_toe
            << ',' << right_toe
            << ',' << cfg_.model_path.string()
            << ',' << input_names_.at(input_index_)
            << ',' << output_names_.at(output_index_)
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
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_name_ptrs_;
    std::vector<const char*> output_name_ptrs_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    std::vector<size_t> input_sizes_;
    std::vector<size_t> output_sizes_;
    std::vector<std::vector<float>> state_inputs_;
    int input_index_ = 0;
    int output_index_ = 0;
    size_t input_size_ = 0;
    size_t output_size_ = 0;
    int history_frames_ = 1;

    std::ofstream event_stream_;
    std::ofstream frame_stream_;
    int event_count_ = 0;
    int frame_count_ = 0;

    std::deque<std::vector<float>> feature_history_;
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
};

}  // namespace mjlab::diagnostics
