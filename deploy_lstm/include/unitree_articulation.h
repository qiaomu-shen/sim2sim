// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/utils/fk_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iterator>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace unitree
{

template <typename LowStatePtr>
class BaseArticulation : public isaaclab::Articulation
{
public:
    BaseArticulation(LowStatePtr lowstate_)
    : lowstate(lowstate_)
    {
        data.joystick = &lowstate->joystick;
    }

    void update() override
    {
        {
            std::lock_guard<std::mutex> lock(lowstate->mutex_);

            // base_angular_velocity
            for (int i(0); i < 3; i++) {
                data.root_ang_vel_b[i] = lowstate->msg_.imu_state().gyroscope()[i];
            }
            // project_gravity_body
            data.root_quat_w = Eigen::Quaternionf(
                lowstate->msg_.imu_state().quaternion()[0],
                lowstate->msg_.imu_state().quaternion()[1],
                lowstate->msg_.imu_state().quaternion()[2],
                lowstate->msg_.imu_state().quaternion()[3]
            );
            data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;

            // joint positions and velocities
            for (int i(0); i < data.joint_ids_map.size(); i++) {
                data.joint_pos[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].q();
                data.joint_vel[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].dq();
            }
        }

        // ── Compute body-frame foot positions via FK ──
        // We assume the joint order matches the G1 29-DOF convention:
        //   0..5:   left  leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
        //   6..11:  right leg
        //   12..14: waist (yaw, roll, pitch)
        //   15..21: left  arm
        //   22..28: right arm
        if (data.joint_ids_map.size() >= 12) {
            float left_angles[6];
            float right_angles[6];
            for (int j = 0; j < 6; j++) {
                left_angles[j]  = data.joint_pos[j];
                right_angles[j] = data.joint_pos[6 + j];
            }

            auto left_params  = isaaclab::fk::kLeftLegParams();
            auto right_params = isaaclab::fk::kRightLegParams();

            auto left_lm  = isaaclab::fk::legFootLandmarks(left_angles, left_params.data());
            auto right_lm = isaaclab::fk::legFootLandmarks(right_angles, right_params.data());

            data.left_toe_pos_body   = left_lm.toe;
            data.right_toe_pos_body  = right_lm.toe;
            data.left_heel_pos_body  = left_lm.heel;
            data.right_heel_pos_body = right_lm.heel;
        }

        reload_external_depth_frame();
    }

    void configure_depth_camera(
        const std::string& camera_name,
        const std::string& topic,
        int out_width,
        int out_height,
        float min_depth,
        float cutoff_distance,
        bool normalize,
        int rotate_k,
        int stack_length,
        const std::string& bridge_file) override
    {
        std::lock_guard<std::mutex> lock(camera_mutex_);
        camera_name_ = camera_name;
        out_width_ = std::max(1, out_width);
        out_height_ = std::max(1, out_height);
        min_depth_ = std::max(0.0f, min_depth);
        cutoff_distance_ = std::max(min_depth_ + 1e-6f, cutoff_distance);
        normalize_ = normalize;
        rotate_k_ = ((rotate_k % 4) + 4) % 4;
        stack_length_ = std::max(1, stack_length);
        depth_topic_ = topic;
        depth_bridge_file_ = bridge_file;
        has_bridge_write_time_ = false;
        latest_depth_frame_.clear();
        depth_stack_.clear();
        data.camera_frames.erase(camera_name_);
    }

    void reload_external_depth_frame() override
    {
        std::string bridge_file;
        {
            std::lock_guard<std::mutex> lock(camera_mutex_);
            bridge_file = depth_bridge_file_;
        }
        if (bridge_file.empty()) {
            return;
        }

        struct stat st {};
        if (stat(bridge_file.c_str(), &st) != 0) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(camera_mutex_);
            if (has_bridge_write_time_ &&
                st.st_size == last_bridge_size_ &&
                st.st_mtim.tv_sec == last_bridge_mtime_.tv_sec &&
                st.st_mtim.tv_nsec == last_bridge_mtime_.tv_nsec) {
                return;
            }
        }

        std::ifstream input(bridge_file, std::ios::binary);
        if (!input) {
            return;
        }

        std::string header;
        if (!std::getline(input, header)) {
            return;
        }

        std::istringstream header_stream(header);
        std::string magic;
        int src_width = 0;
        int src_height = 0;
        int step = 0;
        int is_bigendian = 0;
        header_stream >> magic >> src_width >> src_height >> step >> is_bigendian;
        if (magic != "UTDEPTH1" || src_width <= 0 || src_height <= 0 || step <= 0) {
            return;
        }

        std::vector<uint8_t> image(
            (std::istreambuf_iterator<char>(input)),
            std::istreambuf_iterator<char>());
        const size_t min_bytes = static_cast<size_t>(step) * static_cast<size_t>(src_height);
        if (image.size() < min_bytes) {
            return;
        }

        push_depth_image_16uc1(image, src_width, src_height, step, is_bigendian != 0);

        {
            std::lock_guard<std::mutex> lock(camera_mutex_);
            last_bridge_mtime_ = st.st_mtim;
            last_bridge_size_ = st.st_size;
            has_bridge_write_time_ = true;
        }
    }

    LowStatePtr lowstate;

private:
    void push_depth_image_16uc1(
        const std::vector<uint8_t>& image,
        int src_width,
        int src_height,
        int step,
        bool is_bigendian)
    {
        if (src_width <= 0 || src_height <= 0 || step <= 0) {
            return;
        }

        std::vector<float> frame(out_width_ * out_height_, 0.0f);
        for (int y = 0; y < out_height_; ++y) {
            for (int x = 0; x < out_width_; ++x) {
                int src_x = 0;
                int src_y = 0;
                output_pixel_to_source(x, y, src_width, src_height, src_x, src_y);
                float depth = depth_from_16uc1(image, step, is_bigendian, src_x, src_y);
                if (!std::isfinite(depth)) {
                    depth = cutoff_distance_;
                }

                depth = std::clamp(depth, min_depth_, cutoff_distance_);
                if (normalize_) {
                    depth = depth / cutoff_distance_;
                    depth = std::clamp(depth, 0.0f, 1.0f);
                }
                frame[y * out_width_ + x] = depth;
            }
        }

        std::lock_guard<std::mutex> lock(camera_mutex_);
        latest_depth_frame_ = std::move(frame);
        if (depth_stack_.empty()) {
            for (int i = 0; i < stack_length_; ++i) {
                depth_stack_.push_back(latest_depth_frame_);
            }
        } else {
            depth_stack_.push_front(latest_depth_frame_);
            while (static_cast<int>(depth_stack_.size()) > stack_length_) {
                depth_stack_.pop_back();
            }
            while (static_cast<int>(depth_stack_.size()) < stack_length_) {
                depth_stack_.push_back(latest_depth_frame_);
            }
        }

        std::vector<float> stacked;
        stacked.reserve(static_cast<size_t>(stack_length_) * latest_depth_frame_.size());
        for (const auto& stacked_frame : depth_stack_) {
            stacked.insert(stacked.end(), stacked_frame.begin(), stacked_frame.end());
        }
        data.camera_frames[camera_name_] = std::move(stacked);
    }

    float depth_from_16uc1(
        const std::vector<uint8_t>& image,
        int step,
        bool is_bigendian,
        int src_x,
        int src_y) const
    {
        const size_t idx = static_cast<size_t>(src_y) * static_cast<size_t>(step) +
                           static_cast<size_t>(src_x) * 2;
        if (idx + 1 >= image.size()) {
            return cutoff_distance_;
        }

        uint16_t raw = 0;
        if (is_bigendian) {
            raw = (static_cast<uint16_t>(image[idx]) << 8) |
                  static_cast<uint16_t>(image[idx + 1]);
        } else {
            raw = static_cast<uint16_t>(image[idx]) |
                  (static_cast<uint16_t>(image[idx + 1]) << 8);
        }

        // RealSense ROS 16UC1 depth images are conventionally millimeters.
        return static_cast<float>(raw) * 0.001f;
    }

    void output_pixel_to_source(
        int out_x,
        int out_y,
        int src_width,
        int src_height,
        int& src_x,
        int& src_y) const
    {
        const int rotated_width = (rotate_k_ % 2 == 0) ? src_width : src_height;
        const int rotated_height = (rotate_k_ % 2 == 0) ? src_height : src_width;

        int rot_x = static_cast<int>((static_cast<float>(out_x) + 0.5f) *
                                     static_cast<float>(rotated_width) /
                                     static_cast<float>(out_width_));
        int rot_y = static_cast<int>((static_cast<float>(out_y) + 0.5f) *
                                     static_cast<float>(rotated_height) /
                                     static_cast<float>(out_height_));
        rot_x = std::clamp(rot_x, 0, rotated_width - 1);
        rot_y = std::clamp(rot_y, 0, rotated_height - 1);

        switch (rotate_k_) {
            case 1:
                src_x = rot_y;
                src_y = src_height - 1 - rot_x;
                break;
            case 2:
                src_x = src_width - 1 - rot_x;
                src_y = src_height - 1 - rot_y;
                break;
            case 3:
                src_x = src_width - 1 - rot_y;
                src_y = rot_x;
                break;
            case 0:
            default:
                src_x = rot_x;
                src_y = rot_y;
                break;
        }

        src_x = std::clamp(src_x, 0, src_width - 1);
        src_y = std::clamp(src_y, 0, src_height - 1);
    }

    std::mutex camera_mutex_;
    std::string camera_name_ = "front_depth";
    std::string depth_topic_;
    std::string depth_bridge_file_;
    int out_width_ = 64;
    int out_height_ = 36;
    int stack_length_ = 8;
    float min_depth_ = 0.01f;
    float cutoff_distance_ = 3.0f;
    bool normalize_ = true;
    int rotate_k_ = 0;
    bool has_bridge_write_time_ = false;
    struct timespec last_bridge_mtime_ {};
    off_t last_bridge_size_ = 0;
    std::vector<float> latest_depth_frame_;
    std::deque<std::vector<float>> depth_stack_;
};

}
