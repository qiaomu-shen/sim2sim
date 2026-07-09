// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// Forward kinematics utilities for computing body-frame foot positions
// from joint encoder readings.  These are needed by the deployable
// stair_latent observation that powers the SlowLatent actor.
//
// Kinematic parameters are derived from the Unitree G1 29-DOF MJCF
// (src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml).

#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace isaaclab {
namespace fk {

// =========================================================================
// G1 leg kinematic parameters
// =========================================================================

// Each link's parent-to-child translation and joint axis, all expressed
// in the parent body frame.
struct LegLinkParams {
  Eigen::Vector3f translation;  // parent → child origin in parent frame
  Eigen::Vector3f axis;         // joint rotation axis in parent frame
  Eigen::Quaternionf fixed_rotation;  // MJCF body quat at zero joint angle
};

// Left-leg kinematic chain from pelvis to ankle_roll_link (6 joints).
// Indices in joint_pos: 0=hip_pitch, 1=hip_roll, 2=hip_yaw,
//                       3=knee, 4=ankle_pitch, 5=ankle_roll.
static constexpr int G1_LEFT_LEG_JOINT_COUNT = 6;
static constexpr int G1_LEFT_LEG_JOINT_START = 0;

// Right-leg kinematic chain (6 joints).
// Indices in joint_pos: 6=hip_pitch, 7=hip_roll, 8=hip_yaw,
//                       9=knee, 10=ankle_pitch, 11=ankle_roll.
static constexpr int G1_RIGHT_LEG_JOINT_COUNT = 6;
static constexpr int G1_RIGHT_LEG_JOINT_START = 6;

// Waist joints (3), for reference: 12=waist_yaw, 13=waist_roll, 14=waist_pitch.
// Arms: 15-21 left, 22-28 right.

inline const std::array<LegLinkParams, G1_LEFT_LEG_JOINT_COUNT>
kLeftLegParams() {
  // Axis convention follows MJCF unit vectors.
  return {{
    // pelvis → left_hip_pitch
    {
      Eigen::Vector3f(0.0f,   0.064452f, -0.1027f),
      Eigen::Vector3f(0.0f, 1.0f, 0.0f),          // Y
      Eigen::Quaternionf::Identity()
    },
    // left_hip_pitch → left_hip_roll
    {
      Eigen::Vector3f(0.0f,   0.052f,    -0.030465f),
      Eigen::Vector3f(1.0f, 0.0f, 0.0f),          // X
      Eigen::Quaternionf(0.996179f, 0.0f, -0.0873386f, 0.0f)
    },
    // left_hip_roll → left_hip_yaw
    {
      Eigen::Vector3f(0.025001f, 0.0f, -0.12412f),
      Eigen::Vector3f(0.0f, 0.0f, 1.0f),          // Z
      Eigen::Quaternionf::Identity()
    },
    // left_hip_yaw → left_knee
    {
      Eigen::Vector3f(-0.078273f, 0.0021489f, -0.17734f),
      Eigen::Vector3f(0.0f, 1.0f, 0.0f),          // Y
      Eigen::Quaternionf(0.996179f, 0.0f, 0.0873386f, 0.0f)
    },
    // left_knee → left_ankle_pitch
    {
      Eigen::Vector3f(0.0f, -9.4445e-05f, -0.30001f),
      Eigen::Vector3f(0.0f, 1.0f, 0.0f),          // Y
      Eigen::Quaternionf::Identity()
    },
    // left_ankle_pitch → left_ankle_roll
    {
      Eigen::Vector3f(0.0f, 0.0f, -0.017558f),
      Eigen::Vector3f(1.0f, 0.0f, 0.0f),          // X
      Eigen::Quaternionf::Identity()
    },
  }};
}

inline const std::array<LegLinkParams, G1_RIGHT_LEG_JOINT_COUNT>
kRightLegParams() {
  return {{
    // pelvis → right_hip_pitch  (Y sign flipped vs left)
    {
      Eigen::Vector3f(0.0f,   -0.064452f, -0.1027f),
      Eigen::Vector3f(0.0f, 1.0f, 0.0f),          // Y
      Eigen::Quaternionf::Identity()
    },
    // right_hip_pitch → right_hip_roll  (Y sign flipped)
    {
      Eigen::Vector3f(0.0f,   -0.052f,    -0.030465f),
      Eigen::Vector3f(1.0f, 0.0f, 0.0f),          // X
      Eigen::Quaternionf(0.996179f, 0.0f, -0.0873386f, 0.0f)
    },
    // right_hip_roll → right_hip_yaw
    {
      Eigen::Vector3f(0.025001f, 0.0f, -0.12412f),
      Eigen::Vector3f(0.0f, 0.0f, 1.0f),          // Z
      Eigen::Quaternionf::Identity()
    },
    // right_hip_yaw → right_knee  (Y sign flipped)
    {
      Eigen::Vector3f(-0.078273f, -0.0021489f, -0.17734f),
      Eigen::Vector3f(0.0f, 1.0f, 0.0f),          // Y
      Eigen::Quaternionf(0.996179f, 0.0f, 0.0873386f, 0.0f)
    },
    // right_knee → right_ankle_pitch  (Y sign flipped)
    {
      Eigen::Vector3f(0.0f, 9.4445e-05f, -0.30001f),
      Eigen::Vector3f(0.0f, 1.0f, 0.0f),          // Y
      Eigen::Quaternionf::Identity()
    },
    // right_ankle_pitch → right_ankle_roll
    {
      Eigen::Vector3f(0.0f, 0.0f, -0.017558f),
      Eigen::Vector3f(1.0f, 0.0f, 0.0f),          // X
      Eigen::Quaternionf::Identity()
    },
  }};
}

// Toe / heel offsets expressed in the ankle_roll_link body frame.
// From g1.xml: left_toe pos="0.12 0 -0.03", left_heel pos="-0.05 0 -0.03"
inline Eigen::Vector3f kToeOffset() {
  return Eigen::Vector3f(0.12f, 0.0f, -0.03f);
}
inline Eigen::Vector3f kHeelOffset() {
  return Eigen::Vector3f(-0.05f, 0.0f, -0.03f);
}

// =========================================================================
// Forward kinematics
// =========================================================================

// Return the 4×4 transform produced by a translation + rotation about axis.
inline Eigen::Matrix4f jointTransform(
    const Eigen::Vector3f& translation,
    const Eigen::Vector3f& axis,
    const Eigen::Quaternionf& fixed_rotation,
    float angle) {
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T.block<3, 3>(0, 0) =
      fixed_rotation.normalized().toRotationMatrix()
      * Eigen::AngleAxisf(angle, axis.normalized()).toRotationMatrix();
  T.block<3, 1>(0, 3) = translation;
  return T;
}

// Compute the body-frame position of a point given the joint angles for
// one leg.  The result is the 3D position in the pelvis (body) frame.
//
// joint_angles:    6 joint angles in order hip_pitch, hip_roll, hip_yaw,
//                   knee, ankle_pitch, ankle_roll.
// link_params:     kinematic parameters for the 6 joints.
// offset_in_last:  toe or heel offset expressed in ankle_roll frame.
inline Eigen::Vector3f legFK(
    const float          joint_angles[6],
    const LegLinkParams  link_params[6],
    const Eigen::Vector3f& offset_in_last) {
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  for (int i = 0; i < 6; ++i) {
    T = T * jointTransform(link_params[i].translation,
                           link_params[i].axis,
                           link_params[i].fixed_rotation,
                           joint_angles[i]);
  }
  // The ankle_roll_link origin is at the last transform's translation.
  // Apply the toe/heel offset in the ankle_roll orientation.
  Eigen::Vector4f pt_local(offset_in_last.x(),
                           offset_in_last.y(),
                           offset_in_last.z(),
                           1.0f);
  Eigen::Vector4f pt_body = T * pt_local;
  return pt_body.head<3>();
}

// Convenience: compute all four foot landmarks for a single leg.
struct FootLandmarks {
  Eigen::Vector3f toe;
  Eigen::Vector3f heel;
};

inline FootLandmarks legFootLandmarks(
    const float          joint_angles[6],
    const LegLinkParams  link_params[6]) {
  FootLandmarks lm;
  lm.toe = legFK(joint_angles, link_params, kToeOffset());
  lm.heel = legFK(joint_angles, link_params, kHeelOffset());
  return lm;
}

}  // namespace fk
}  // namespace isaaclab
