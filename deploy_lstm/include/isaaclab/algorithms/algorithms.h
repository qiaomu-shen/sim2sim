// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace isaaclab
{

class Algorithms
{
public:
    virtual ~Algorithms() = default;
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;
    virtual void reset() {}

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_name_storage.push_back(input_name.get());
        }

        for (const auto& shape : input_shapes) {
            input_sizes.push_back(tensor_size(shape));
        }

        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            Ort::TypeInfo output_type = session->GetOutputTypeInfo(i);
            output_shapes.push_back(output_type.GetTensorTypeAndShapeInfo().GetShape());
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            output_name_storage.push_back(output_name.get());
        }

        for (const auto& shape : output_shapes) {
            output_sizes.push_back(tensor_size(shape));
        }

        refresh_name_ptrs();

        action_output_idx = find_output("actions");
        if (action_output_idx < 0) {
            action_output_idx = 0;
        }

        h_input_idx = find_input("h_in");
        c_input_idx = find_input("c_in");
        h_output_idx = find_output("h_out");
        c_output_idx = find_output("c_out");

        // ── Slow-latent extra state I/O ──
        z_input_idx  = find_input("z_in");
        z_output_idx = find_output("z_out");
        gate_input_idx  = find_input("gate_state_in");
        gate_output_idx = find_output("gate_state_out");

        recurrent = h_input_idx >= 0 || c_input_idx >= 0
                 || h_output_idx >= 0 || c_output_idx >= 0
                 || z_input_idx >= 0 || z_output_idx >= 0
                 || gate_input_idx >= 0 || gate_output_idx >= 0;

        if (recurrent) {
            if (h_input_idx >= 0) {
                h_state.resize(input_sizes[h_input_idx], 0.0f);
            }
            if (c_input_idx >= 0) {
                c_state.resize(input_sizes[c_input_idx], 0.0f);
            }
            if (z_input_idx >= 0) {
                z_state.resize(input_sizes[z_input_idx], 0.0f);
            }
            if (gate_input_idx >= 0) {
                gate_state.resize(input_sizes[gate_input_idx], 0.0f);
            }
        }

        action.resize(output_sizes[action_output_idx], 0.0f);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(size_t i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            if (name_str == "h_in") {
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    memory_info, h_state.data(), h_state.size(), input_shapes[i].data(), input_shapes[i].size()));
            } else if (name_str == "c_in") {
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    memory_info, c_state.data(), c_state.size(), input_shapes[i].data(), input_shapes[i].size()));
            } else if (name_str == "z_in") {
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    memory_info, z_state.data(), z_state.size(), input_shapes[i].data(), input_shapes[i].size()));
            } else if (name_str == "gate_state_in") {
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    memory_info, gate_state.data(), gate_state.size(), input_shapes[i].data(), input_shapes[i].size()));
            } else {
                if (obs.find(name_str) == obs.end()) {
                    throw std::runtime_error("Input name " + name_str + " not found in observations.");
                }
                auto& input_data = obs.at(name_str);
                if (input_data.size() != input_sizes[i]) {
                    throw std::runtime_error(
                        "Input " + name_str + " size mismatch. Expected " +
                        std::to_string(input_sizes[i]) + ", got " + std::to_string(input_data.size()) + ".");
                }
                const auto invalid = std::find_if(
                    input_data.begin(), input_data.end(),
                    [](float value) { return !std::isfinite(value); });
                if (invalid != input_data.end()) {
                    return fail_safe_zero_action(
                        "non-finite value in ONNX input " + name_str + " at index " +
                        std::to_string(std::distance(input_data.begin(), invalid)));
                }
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    memory_info, input_data.data(), input_data.size(), input_shapes[i].data(), input_shapes[i].size()));
            }
        }

        // Run the model
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size());

        // Copy output data
        auto action_data = output_tensors[action_output_idx].GetTensorMutableData<float>();
        for (size_t i = 0; i < action.size(); ++i) {
            if (!std::isfinite(action_data[i])) {
                return fail_safe_zero_action(
                    "non-finite ONNX action at index " + std::to_string(i));
            }
        }
        {
            std::lock_guard<std::mutex> lock(act_mtx_);
            std::memcpy(action.data(), action_data, action.size() * sizeof(float));
        }

        if (recurrent) {
            if (h_output_idx >= 0 && !h_state.empty()) {
                auto h_out = output_tensors[h_output_idx].GetTensorMutableData<float>();
                std::copy(h_out, h_out + h_state.size(), h_state.begin());
            }
            if (c_output_idx >= 0 && !c_state.empty()) {
                auto c_out = output_tensors[c_output_idx].GetTensorMutableData<float>();
                std::copy(c_out, c_out + c_state.size(), c_state.begin());
            }
            if (z_output_idx >= 0 && !z_state.empty()) {
                auto z_out = output_tensors[z_output_idx].GetTensorMutableData<float>();
                std::copy(z_out, z_out + z_state.size(), z_state.begin());
            }
            if (gate_output_idx >= 0 && !gate_state.empty()) {
                auto g_out = output_tensors[gate_output_idx].GetTensorMutableData<float>();
                std::copy(g_out, g_out + gate_state.size(), gate_state.begin());
            }
        }
        return action;
    }

    void reset() override
    {
        std::fill(h_state.begin(), h_state.end(), 0.0f);
        std::fill(c_state.begin(), c_state.end(), 0.0f);
        std::fill(z_state.begin(), z_state.end(), 0.0f);
        std::fill(gate_state.begin(), gate_state.end(), 0.0f);
    }

#ifdef MJLAB_ORT_RUNNER_TESTING
    size_t recurrent_nonzero_count_for_testing() const
    {
        return nonzero_count(h_state) +
               nonzero_count(c_state) +
               nonzero_count(z_state) +
               nonzero_count(gate_state);
    }
#endif

private:
    static size_t nonzero_count(const std::vector<float>& values)
    {
        return static_cast<size_t>(std::count_if(
            values.begin(),
            values.end(),
            [](float value) { return std::isfinite(value) && std::abs(value) > 1.0e-8f; }));
    }

    std::vector<float> fail_safe_zero_action(const std::string& reason)
    {
        std::cerr << "[OrtRunner safety] " << reason
                  << "; commanding zero raw action and resetting recurrent state.\n";
        reset();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::fill(action.begin(), action.end(), 0.0f);
        return action;
    }

    static size_t tensor_size(const std::vector<int64_t>& shape)
    {
        size_t size = 1;
        for (const auto& dim : shape) {
            if (dim <= 0) {
                throw std::runtime_error("Dynamic ONNX tensor dimensions are not supported by OrtRunner.");
            }
            size *= static_cast<size_t>(dim);
        }
        return size;
    }

    int find_input(const std::string& name) const
    {
        return find_name(input_name_storage, name);
    }

    int find_output(const std::string& name) const
    {
        return find_name(output_name_storage, name);
    }

    static int find_name(const std::vector<std::string>& names, const std::string& name)
    {
        auto it = std::find(names.begin(), names.end(), name);
        if (it == names.end()) {
            return -1;
        }
        return static_cast<int>(std::distance(names.begin(), it));
    }

    void refresh_name_ptrs()
    {
        input_names.clear();
        input_names.reserve(input_name_storage.size());
        for (const auto& name : input_name_storage) {
            input_names.push_back(name.c_str());
        }

        output_names.clear();
        output_names.reserve(output_name_storage.size());
        for (const auto& name : output_name_storage) {
            output_names.push_back(name.c_str());
        }
    }

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_name_storage;
    std::vector<std::string> output_name_storage;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<size_t> input_sizes;
    std::vector<size_t> output_sizes;

    int action_output_idx = 0;
    int h_input_idx = -1;
    int c_input_idx = -1;
    int h_output_idx = -1;
    int c_output_idx = -1;

    // Slow-latent extra state indices
    int z_input_idx = -1;
    int z_output_idx = -1;
    int gate_input_idx = -1;
    int gate_output_idx = -1;

    bool recurrent = false;
    std::vector<float> h_state;
    std::vector<float> c_state;
    std::vector<float> z_state;      // slow latent memory z_t (16 dim)
    std::vector<float> gate_state;   // gate state machine (5 dim)
};
};
