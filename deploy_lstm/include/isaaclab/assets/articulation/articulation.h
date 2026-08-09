// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>
#include "unitree/dds_wrapper/common/unitree_joystick.hpp"

namespace isaaclab
{

class MotionLoader;

struct ArticulationData
{
    Eigen::Vector3f GRAVITY_VEC_W = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    Eigen::Vector3f FORWARD_VEC_B = Eigen::Vector3f(1.0f, 0.0f, 0.0f);

    std::vector<float> joint_stiffness; // sdk order
    std::vector<float> joint_damping; // sdk order

    // Joint positions of all joints.
    Eigen::VectorXf joint_pos;
    
    // Default joint positions of all joints.
    Eigen::VectorXf default_joint_pos;

    // Joint velocities of all joints.
    Eigen::VectorXf joint_vel;

    // Root angular velocity in base world frame.
    Eigen::Vector3f root_ang_vel_b;

    // Projection of the gravity direction on base frame.
    Eigen::Vector3f projected_gravity_b;

    Eigen::Quaternionf root_quat_w;

    std::vector<int> joint_ids_map;

    unitree::common::UnitreeJoystick* joystick = nullptr;

    // ── FK-derived body-frame foot positions (for stair_latent obs) ──
    Eigen::Vector3f left_toe_pos_body    = Eigen::Vector3f::Zero();
    Eigen::Vector3f right_toe_pos_body   = Eigen::Vector3f::Zero();
    Eigen::Vector3f left_heel_pos_body   = Eigen::Vector3f::Zero();
    Eigen::Vector3f right_heel_pos_body  = Eigen::Vector3f::Zero();

    // ── Previous-frame caches for delta features ──
    Eigen::Vector3f prev_left_toe_pos_body  = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_right_toe_pos_body = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_base_ang_vel       = Eigen::Vector3f::Zero();
    Eigen::VectorXf prev_leg_joint_vel;      // 12-DOF leg joint vel (resized on init)

    // ── Previous toe velocity caches (for toe_vel_delta features) ──
    Eigen::Vector3f prev_left_toe_vel   = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_right_toe_vel  = Eigen::Vector3f::Zero();
    bool            prev_toe_vel_valid  = false;

    // ── Previous action cache (full 29-DOF, for last_action obs term) ──
    std::vector<float> prev_action;          // resized to joint_ids_map.size() on init

    // ── stair_latent cache validity flag ──
    bool stair_latent_cache_valid = false;

    // Runtime camera observations keyed by camera name, e.g. "front_depth".
    std::unordered_map<std::string, std::vector<float>> camera_frames;

    // SlowLatent FootEventMemory compact summary.  The policy consumes the
    // 80-dim summary after the 93-dim stair_latent vector.
    std::vector<float> foot_event_summary = std::vector<float>(80, 0.0f);
};

class Articulation
{
public:
    Articulation(){}

    virtual void update(){};

    virtual void configure_depth_camera(
        const std::string& camera_name,
        const std::string& topic,
        int out_width,
        int out_height,
        float min_depth,
        float cutoff_distance,
        bool normalize,
        int rotate_k,
        int stack_length,
        const std::string& bridge_file) {}

    virtual void reload_external_depth_frame() {}

    ArticulationData data;
};

};
