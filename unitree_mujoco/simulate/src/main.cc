// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// !!! hack code: make glfw_adapter.window_ public
#define private public
#include "glfw_adapter.h"
#undef private

#include <chrono>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include <mujoco/mujoco.h>
#include "simulate.h"
#include "array_safety.h"
#include "unitree_sdk2_bridge.h"
#include "param.h"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"
#define NUM_MOTOR_IDL_GO 20

extern "C"
{
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif
}

class ElasticBand
{
public:
  ElasticBand(){};
  void Advance(std::vector<double> x, std::vector<double> dx)
  {
    std::vector<double> delta_x = {0.0, 0.0, 0.0};
    delta_x[0] = point_[0] - x[0];
    delta_x[1] = point_[1] - x[1];
    delta_x[2] = point_[2] - x[2];
    double distance = sqrt(delta_x[0] * delta_x[0] + delta_x[1] * delta_x[1] + delta_x[2] * delta_x[2]);
    if (distance < 1.0e-6)
    {
      f_[0] = 0.0;
      f_[1] = 0.0;
      f_[2] = 0.0;
      return;
    }

    std::vector<double> direction = {0.0, 0.0, 0.0};
    direction[0] = delta_x[0] / distance;
    direction[1] = delta_x[1] / distance;
    direction[2] = delta_x[2] / distance;

    double v = dx[0] * direction[0] + dx[1] * direction[1] + dx[2] * direction[2];

    f_[0] = (stiffness_ * (distance - length_) - damping_ * v) * direction[0];
    f_[1] = (stiffness_ * (distance - length_) - damping_ * v) * direction[1];
    f_[2] = (stiffness_ * (distance - length_) - damping_ * v) * direction[2];
  }


  double stiffness_ = 200;
  double damping_ = 100;
  std::vector<double> point_ = {0, 0, 3};
  double length_ = 0.0;
  bool enable_ = true;
  std::vector<double> f_ = {0, 0, 0};
};
inline ElasticBand elastic_band;


namespace
{
  namespace mj = ::mujoco;
  namespace mju = ::mujoco::sample_util;

  // constants
  const double syncMisalign = 0.1;       // maximum mis-alignment before re-sync (simulation seconds)
  const double simRefreshFraction = 0.7; // fraction of refresh available for simulation
  const int kErrorLength = 1024;         // load error string length

  // model and data
  mjModel *m = nullptr;
  mjData *d = nullptr;

  // control noise variables
  mjtNum *ctrlnoise = nullptr;

  using Seconds = std::chrono::duration<double>;

  double NormalizeAngle(double angle)
  {
    while (angle > mjPI)
    {
      angle -= 2.0 * mjPI;
    }
    while (angle < -mjPI)
    {
      angle += 2.0 * mjPI;
    }
    return angle;
  }

  void ApplyElasticBandAutoAlign(mjModel* model, mjData* data, const int body_id)
  {
    if (!model || !data || param::config.elastic_band_auto_align == 0 ||
        body_id < 0 || body_id >= model->nbody)
    {
      return;
    }

    const mjtNum* xmat = data->xmat + 9 * body_id;
    const mjtNum* cvel = data->cvel + 6 * body_id;
    const double torque_limit =
      std::max(0.0, param::config.elastic_band_align_torque_limit);

    const std::array<double, 3> body_z = {
      static_cast<double>(xmat[2]),
      static_cast<double>(xmat[5]),
      static_cast<double>(xmat[8])
    };
    const double kp_upright = param::config.elastic_band_align_kp_upright;
    const double kd_upright = param::config.elastic_band_align_kd_upright;

    std::array<double, 3> torque = {
      kp_upright * body_z[1] - kd_upright * static_cast<double>(cvel[0]),
      -kp_upright * body_z[0] - kd_upright * static_cast<double>(cvel[1]),
      0.0
    };

    const double yaw = std::atan2(static_cast<double>(xmat[3]),
                                  static_cast<double>(xmat[0]));
    const double target_yaw =
      param::config.elastic_band_target_yaw_deg * mjPI / 180.0;
    const double yaw_error = NormalizeAngle(target_yaw - yaw);
    torque[2] += param::config.elastic_band_align_kp_yaw * yaw_error -
      param::config.elastic_band_align_kd_yaw * static_cast<double>(cvel[2]);

    const int wrench = 6 * body_id;
    for (int i = 0; i < 3; ++i)
    {
      const double value = torque_limit > 0.0
        ? std::clamp(torque[i], -torque_limit, torque_limit)
        : torque[i];
      data->xfrc_applied[wrench + 3 + i] += static_cast<mjtNum>(value);
    }
  }

  std::string MakeRunTimestamp()
  {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
    localtime_r(&time, &local_time);

    char buffer[32] = {};
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &local_time);
    return buffer;
  }

  class TouchdownTruthLogger
  {
  public:
    void MaybeWrite(const mjModel* model, const mjData* data)
    {
      if (!model || !data || param::config.enable_touchdown_truth_log == 0)
      {
        return;
      }

      if (model != model_)
      {
        ResetForModel(model);
      }
      MaybeReopenForLatestRunDir(data);

      const FootContact left = DetectFootContact(model, data, left_foot_body_id_);
      const FootContact right = DetectFootContact(model, data, right_foot_body_id_);
      UpdateFoot("left", left, data);
      UpdateFoot("right", right, data);
      MaybeWriteContactState(left, right, data);
    }

  private:
    struct FootContact
    {
      bool contact = false;
      int contact_count = 0;
      double normal_force = 0.0;
      double pos[3] = {0.0, 0.0, 0.0};
      int geom_id = -1;
      int other_geom_id = -1;
    };

    struct FootState
    {
      bool prev_contact = false;
      bool has_seen_air = false;
      int air_start_tick_ms = -1;
      int last_touchdown_tick_ms = -1000000;
    };

    static bool IsBodyDescendantOf(const mjModel* model, int body_id, int root_id)
    {
      if (!model || body_id < 0 || root_id < 0)
      {
        return false;
      }
      while (body_id >= 0)
      {
        if (body_id == root_id)
        {
          return true;
        }
        if (body_id == 0)
        {
          break;
        }
        body_id = model->body_parentid[body_id];
      }
      return false;
    }

    static const char* SafeName(const mjModel* model, int type, int id)
    {
      if (!model || id < 0)
      {
        return "";
      }
      const char* name = mj_id2name(model, type, id);
      return name ? name : "";
    }

    bool IsEnvironmentBody(const mjModel* model, int body_id) const
    {
      if (body_id == 0)
      {
        return true;
      }
      if (robot_root_body_id_ >= 0)
      {
        return !IsBodyDescendantOf(model, body_id, robot_root_body_id_);
      }
      return false;
    }

    FootContact DetectFootContact(
      const mjModel* model,
      const mjData* data,
      int foot_body_id) const
    {
      FootContact result;
      if (foot_body_id < 0)
      {
        return result;
      }

      for (int i = 0; i < data->ncon; ++i)
      {
        const mjContact& contact = data->contact[i];
        const int geom1 = contact.geom1;
        const int geom2 = contact.geom2;
        if (geom1 < 0 || geom2 < 0)
        {
          continue;
        }

        const int body1 = model->geom_bodyid[geom1];
        const int body2 = model->geom_bodyid[geom2];
        const bool geom1_is_foot = IsBodyDescendantOf(model, body1, foot_body_id);
        const bool geom2_is_foot = IsBodyDescendantOf(model, body2, foot_body_id);
        if (geom1_is_foot == geom2_is_foot)
        {
          continue;
        }

        const int foot_geom = geom1_is_foot ? geom1 : geom2;
        const int other_geom = geom1_is_foot ? geom2 : geom1;
        const int other_body = model->geom_bodyid[other_geom];
        if (!IsEnvironmentBody(model, other_body))
        {
          continue;
        }

        mjtNum force[6] = {0, 0, 0, 0, 0, 0};
        mj_contactForce(model, data, i, force);
        result.contact = true;
        result.contact_count += 1;
        result.normal_force += std::max(0.0, static_cast<double>(force[0]));
        result.pos[0] += contact.pos[0];
        result.pos[1] += contact.pos[1];
        result.pos[2] += contact.pos[2];
        result.geom_id = foot_geom;
        result.other_geom_id = other_geom;
      }

      if (result.contact_count > 0)
      {
        result.pos[0] /= result.contact_count;
        result.pos[1] /= result.contact_count;
        result.pos[2] /= result.contact_count;
      }
      return result;
    }

    void ResetForModel(const mjModel* model)
    {
      Close();
      model_ = model;
      left_ = FootState{};
      right_ = FootState{};
      event_count_ = 0;
      last_state_tick_ms_ = -1;
      last_latest_check_time_s_ = -1.0;
      left_foot_body_id_ = mj_name2id(model, mjOBJ_BODY, "left_ankle_roll_link");
      right_foot_body_id_ = mj_name2id(model, mjOBJ_BODY, "right_ankle_roll_link");
      robot_root_body_id_ = mj_name2id(model, mjOBJ_BODY, "pelvis");
      if (robot_root_body_id_ < 0)
      {
        robot_root_body_id_ = mj_name2id(model, mjOBJ_BODY, "pelvis_link");
      }
      Open();
    }

    static std::filesystem::path ReadPathFile(const std::filesystem::path& path)
    {
      std::ifstream stream(path);
      std::string line;
      if (!stream || !std::getline(stream, line) || line.empty())
      {
        return {};
      }
      return std::filesystem::path(line);
    }

    std::filesystem::path FallbackRunDir()
    {
      if (!fallback_run_dir_.empty())
      {
        return fallback_run_dir_;
      }

      const std::filesystem::path configured(param::config.touchdown_truth_log_file);
      std::filesystem::path root = configured.parent_path();
      if (root.empty())
      {
        root = ".";
      }

      fallback_run_dir_ = root / MakeRunTimestamp();
      int suffix = 2;
      while (std::filesystem::exists(fallback_run_dir_))
      {
        fallback_run_dir_ =
          root / (MakeRunTimestamp() + "_" + std::to_string(suffix));
        suffix += 1;
      }
      return fallback_run_dir_;
    }

    std::filesystem::path ResolveLogPath()
    {
      const std::filesystem::path configured(param::config.touchdown_truth_log_file);
      const std::filesystem::path filename =
        configured.filename().empty()
          ? std::filesystem::path("touchdown_truth.csv")
          : configured.filename();

      if (param::config.touchdown_truth_use_latest_run_dir != 0)
      {
        const std::filesystem::path latest_file(
          param::config.touchdown_truth_latest_run_dir_file);
        const std::filesystem::path latest_run_dir = ReadPathFile(latest_file);
        if (!latest_run_dir.empty())
        {
          return latest_run_dir / filename;
        }
      }

      if (param::config.touchdown_truth_timestamp_run_dir != 0)
      {
        return FallbackRunDir() / filename;
      }
      return configured;
    }

    void MaybeReopenForLatestRunDir(const mjData* data)
    {
      if (param::config.touchdown_truth_use_latest_run_dir == 0)
      {
        return;
      }

      if (last_latest_check_time_s_ >= 0.0 &&
          data->time >= last_latest_check_time_s_ &&
          data->time - last_latest_check_time_s_ < 0.1)
      {
        return;
      }
      last_latest_check_time_s_ = data->time;

      const std::filesystem::path path = ResolveLogPath();
      if (path.empty() || path == active_log_path_)
      {
        return;
      }
      Open(path);
    }

    void Open()
    {
      Open(ResolveLogPath());
    }

    void Open(const std::filesystem::path& path)
    {
      if (path.empty())
      {
        return;
      }
      if (stream_.is_open() && path == active_log_path_)
      {
        return;
      }
      Close();
      if (path.has_parent_path())
      {
        std::filesystem::create_directories(path.parent_path());
      }
      stream_.open(path, std::ios::out | std::ios::trunc);
      if (!stream_)
      {
        std::cerr << "Failed to open touchdown truth log: "
                  << path.string() << std::endl;
        return;
      }
      stream_.setf(std::ios::fixed);
      stream_ << std::setprecision(7);
      stream_
        << "event_type,foot,tick_ms,time_s,contact_count,normal_force"
        << ",contact_pos_w_x,contact_pos_w_y,contact_pos_w_z"
        << ",foot_geom_id,foot_geom_name,other_geom_id,other_geom_name\n";
      stream_.flush();
      active_log_path_ = path;
      left_ = FootState{};
      right_ = FootState{};
      event_count_ = 0;
      last_state_tick_ms_ = -1;
      std::cout << "Touchdown truth log enabled: "
                << path.string() << std::endl;
    }

    void Close()
    {
      if (stream_.is_open())
      {
        stream_.flush();
        stream_.close();
      }
      active_log_path_.clear();
    }

    void UpdateFoot(
      const std::string& foot,
      const FootContact& contact,
      const mjData* data)
    {
      FootState& state = foot == "left" ? left_ : right_;
      const int tick_ms = TickMs(data);
      if (!contact.contact)
      {
        state.has_seen_air = true;
        if (state.air_start_tick_ms < 0 || state.prev_contact)
        {
          state.air_start_tick_ms = tick_ms;
        }
      }

      if (!contact.contact && state.prev_contact)
      {
        WriteRow("sim_liftoff", foot, contact, data);
      }

      const int air_time_ms =
        state.air_start_tick_ms >= 0 ? tick_ms - state.air_start_tick_ms : 0;
      const bool enough_air =
        air_time_ms >= param::config.touchdown_truth_min_air_ms;
      const bool enough_cooldown =
        tick_ms - state.last_touchdown_tick_ms >=
        param::config.touchdown_truth_cooldown_ms;

      if (contact.contact && !state.prev_contact && state.has_seen_air &&
          enough_air && enough_cooldown)
      {
        WriteRow("sim_touchdown", foot, contact, data);
        state.last_touchdown_tick_ms = tick_ms;
      }
      state.prev_contact = contact.contact;
    }

    void MaybeWriteContactState(
      const FootContact& left,
      const FootContact& right,
      const mjData* data)
    {
      const int interval_ms =
        param::config.touchdown_truth_state_interval_ms;
      if (interval_ms <= 0)
      {
        return;
      }

      const int tick_ms = TickMs(data);
      if (last_state_tick_ms_ >= 0 && tick_ms - last_state_tick_ms_ < interval_ms)
      {
        return;
      }

      WriteRow("contact_state", "left", left, data);
      WriteRow("contact_state", "right", right, data);
      last_state_tick_ms_ = tick_ms;
    }

    static int TickMs(const mjData* data)
    {
      return static_cast<int>(std::round(data->time / 1.0e-3));
    }

    void WriteRow(
      const std::string& event_type,
      const std::string& foot,
      const FootContact& contact,
      const mjData* data)
    {
      if (!stream_.is_open() || !model_)
      {
        return;
      }
      const int tick_ms = TickMs(data);
      stream_
        << event_type
        << ',' << foot
        << ',' << tick_ms
        << ',' << data->time
        << ',' << contact.contact_count
        << ',' << contact.normal_force
        << ',' << contact.pos[0]
        << ',' << contact.pos[1]
        << ',' << contact.pos[2]
        << ',' << contact.geom_id
        << ',' << SafeName(model_, mjOBJ_GEOM, contact.geom_id)
        << ',' << contact.other_geom_id
        << ',' << SafeName(model_, mjOBJ_GEOM, contact.other_geom_id)
        << '\n';

      event_count_ += 1;
      const int flush_interval =
        std::max(1, param::config.touchdown_truth_flush_interval_events);
      if (event_count_ % flush_interval == 0)
      {
        stream_.flush();
      }
    }

    const mjModel* model_ = nullptr;
    int left_foot_body_id_ = -1;
    int right_foot_body_id_ = -1;
    int robot_root_body_id_ = -1;
    FootState left_;
    FootState right_;
    std::ofstream stream_;
    std::filesystem::path active_log_path_;
    std::filesystem::path fallback_run_dir_;
    int event_count_ = 0;
    int last_state_tick_ms_ = -1;
    double last_latest_check_time_s_ = -1.0;
  };

  class DepthBridge
  {
  public:
    void MaybeWrite(const mjModel* model, const mjData* data)
    {
      if (!model || !data || param::config.enable_depth_bridge == 0)
      {
        return;
      }

      if (model != model_)
      {
        model_ = model;
        body_id_ = ResolveCameraBody(model);
        last_write_time_ = -1.0e9;
      }

      const double fps = std::max(1.0, param::config.depth_fps);
      if (data->time - last_write_time_ < 1.0 / fps)
      {
        return;
      }

      const int width = std::max(1, param::config.depth_width);
      const int height = std::max(1, param::config.depth_height);
      const double min_depth = std::max(0.0, param::config.depth_min);
      const double cutoff = std::max(min_depth + 1.0e-6, param::config.depth_cutoff);

      std::vector<uint8_t> raw(static_cast<size_t>(width) * static_cast<size_t>(height) * 2u, 0);
      RenderDepth16UC1(model, data, width, height, min_depth, cutoff, raw);
      WriteDepthFile(width, height, raw);

      last_write_time_ = data->time;
    }

  private:
    static int ResolveCameraBody(const mjModel* model)
    {
      int body_id = mj_name2id(model, mjOBJ_BODY, param::config.depth_camera_body.c_str());
      if (body_id >= 0)
      {
        return body_id;
      }

      body_id = mj_name2id(model, mjOBJ_BODY, "torso_link");
      if (body_id >= 0)
      {
        return body_id;
      }

      body_id = mj_name2id(model, mjOBJ_BODY, "base_link");
      if (body_id >= 0)
      {
        return body_id;
      }

      return 0;
    }

    static void TransformLocalVector(const mjtNum* mat, const mjtNum local[3], mjtNum out[3])
    {
      out[0] = mat[0] * local[0] + mat[1] * local[1] + mat[2] * local[2];
      out[1] = mat[3] * local[0] + mat[4] * local[1] + mat[5] * local[2];
      out[2] = mat[6] * local[0] + mat[7] * local[1] + mat[8] * local[2];
    }

    void RenderDepth16UC1(
      const mjModel* model,
      const mjData* data,
      int width,
      int height,
      double min_depth,
      double cutoff,
      std::vector<uint8_t>& raw) const
    {
      const mjtNum* xpos = data->xpos + 3 * body_id_;
      const mjtNum* xmat = data->xmat + 9 * body_id_;

      mjtNum local_pos[3] = {
        static_cast<mjtNum>(param::config.depth_pos_x),
        static_cast<mjtNum>(param::config.depth_pos_y),
        static_cast<mjtNum>(param::config.depth_pos_z)
      };

      mjtNum world_offset[3];
      TransformLocalVector(xmat, local_pos, world_offset);

      mjtNum origin[3] = {
        xpos[0] + world_offset[0],
        xpos[1] + world_offset[1],
        xpos[2] + world_offset[2]
      };

      const double pitch = param::config.depth_pitch_deg * mjPI / 180.0;
      // Match MuJoCo camera convention used by training:
      // camera +X is image right, +Y is image up, and -Z is the optical axis.
      // depth_pitch_deg is the optical-axis elevation in the torso x-z plane
      // (negative means pitched downward).
      const mjtNum local_forward[3] = {
        static_cast<mjtNum>(std::cos(pitch)),
        0,
        static_cast<mjtNum>(std::sin(pitch))
      };
      const mjtNum local_right[3] = {
        static_cast<mjtNum>(-std::sin(pitch)),
        0,
        static_cast<mjtNum>(std::cos(pitch))
      };
      const mjtNum local_up[3] = {0, 1, 0};

      mjtNum forward[3];
      mjtNum right[3];
      mjtNum up[3];
      TransformLocalVector(xmat, local_forward, forward);
      TransformLocalVector(xmat, local_right, right);
      TransformLocalVector(xmat, local_up, up);

      const double fovy = std::clamp(param::config.depth_fovy, 1.0, 170.0) * mjPI / 180.0;
      const double tan_y = std::tan(0.5 * fovy);
      const double tan_x = tan_y * static_cast<double>(width) / static_cast<double>(height);

      for (int y = 0; y < height; ++y)
      {
        const double v = 1.0 - 2.0 * (static_cast<double>(y) + 0.5) / static_cast<double>(height);
        for (int x = 0; x < width; ++x)
        {
          const double u = 2.0 * (static_cast<double>(x) + 0.5) / static_cast<double>(width) - 1.0;

          mjtNum ray[3] = {
            static_cast<mjtNum>(forward[0] + right[0] * u * tan_x + up[0] * v * tan_y),
            static_cast<mjtNum>(forward[1] + right[1] * u * tan_x + up[1] * v * tan_y),
            static_cast<mjtNum>(forward[2] + right[2] * u * tan_x + up[2] * v * tan_y)
          };
          mju_normalize3(ray);

          int geom_id = -1;
          mjtNum distance = mj_ray(model, data, origin, ray, nullptr, 1, body_id_, &geom_id);
          const double axial_scale =
            static_cast<double>(ray[0] * forward[0] + ray[1] * forward[1] + ray[2] * forward[2]);
          double depth = (distance > 0 && axial_scale > 1.0e-6)
            ? static_cast<double>(distance) * axial_scale
            : cutoff;
          depth = std::clamp(depth, min_depth, cutoff);

          const uint16_t depth_mm = static_cast<uint16_t>(
            std::clamp(std::lround(depth * 1000.0), 0l, 65535l));
          const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width) +
                              static_cast<size_t>(x)) * 2u;
          raw[idx] = static_cast<uint8_t>(depth_mm & 0xffu);
          raw[idx + 1] = static_cast<uint8_t>((depth_mm >> 8) & 0xffu);
        }
      }
    }

    static void WriteDepthFile(int width, int height, const std::vector<uint8_t>& raw)
    {
      const std::string& output = param::config.depth_bridge_file;
      if (output.empty())
      {
        return;
      }

      const std::string tmp = output + ".mujoco.tmp";
      std::ofstream file(tmp, std::ios::binary);
      if (!file)
      {
        return;
      }

      file << "UTDEPTH1 " << width << " " << height << " " << width * 2 << " 0\n";
      file.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
      file.close();

      std::rename(tmp.c_str(), output.c_str());
    }

    const mjModel* model_ = nullptr;
    int body_id_ = 0;
    double last_write_time_ = -1.0e9;
  };

  class HeightScanBridge
  {
  public:
    void MaybeWrite(const mjModel* model, const mjData* data)
    {
      if (!model || !data || param::config.enable_height_scan_bridge == 0)
      {
        return;
      }

      if (model != model_)
      {
        model_ = model;
        body_id_ = ResolveBody(model);
        last_write_time_ = -1.0e9;
      }

      const double fps = std::max(1.0, param::config.height_scan_fps);
      if (data->time - last_write_time_ < 1.0 / fps)
      {
        return;
      }

      std::vector<float> values;
      RenderHeightScan(model, data, values);
      WriteHeightScanFile(values);

      last_write_time_ = data->time;
    }

  private:
    static int ResolveBody(const mjModel* model)
    {
      int body_id = mj_name2id(model, mjOBJ_BODY, param::config.height_scan_body.c_str());
      if (body_id >= 0)
      {
        return body_id;
      }

      body_id = mj_name2id(model, mjOBJ_BODY, "pelvis");
      if (body_id >= 0)
      {
        return body_id;
      }

      body_id = mj_name2id(model, mjOBJ_BODY, "pelvis_link");
      if (body_id >= 0)
      {
        return body_id;
      }

      return 0;
    }

    static int GridCount(double size, double resolution)
    {
      size = std::max(0.0, size);
      resolution = std::max(1.0e-6, resolution);
      return std::max(1, static_cast<int>(std::floor(size / resolution + 0.5)) + 1);
    }

    void RenderHeightScan(
      const mjModel* model,
      const mjData* data,
      std::vector<float>& values) const
    {
      const int count_x = GridCount(param::config.height_scan_size_x,
                                    param::config.height_scan_resolution);
      const int count_y = GridCount(param::config.height_scan_size_y,
                                    param::config.height_scan_resolution);
      values.clear();
      values.reserve(static_cast<size_t>(count_x) * static_cast<size_t>(count_y));

      const mjtNum* xpos = data->xpos + 3 * body_id_;
      const mjtNum* xmat = data->xmat + 9 * body_id_;

      const double yaw = std::atan2(static_cast<double>(xmat[3]),
                                    static_cast<double>(xmat[0]));
      const double cos_yaw = std::cos(yaw);
      const double sin_yaw = std::sin(yaw);
      const double resolution = std::max(1.0e-6, param::config.height_scan_resolution);
      const double start_x = -0.5 * param::config.height_scan_size_x;
      const double start_y = -0.5 * param::config.height_scan_size_y;
      const double ray_start_z = param::config.height_scan_ray_start_z;
      const double offset = param::config.height_scan_offset;
      const double sensor_height =
        static_cast<double>(xpos[2]) +
        (param::config.height_scan_sensor_height_includes_ray_start_z != 0 ? ray_start_z : 0.0);
      const double value_clip = param::config.height_scan_value_clip;

      for (int iy = 0; iy < count_y; ++iy)
      {
        const double local_y = start_y + static_cast<double>(iy) * resolution;
        for (int ix = 0; ix < count_x; ++ix)
        {
          const double local_x = start_x + static_cast<double>(ix) * resolution;
          mjtNum origin[3] = {
            static_cast<mjtNum>(xpos[0] + cos_yaw * local_x - sin_yaw * local_y),
            static_cast<mjtNum>(xpos[1] + sin_yaw * local_x + cos_yaw * local_y),
            static_cast<mjtNum>(xpos[2] + ray_start_z)
          };
          mjtNum ray[3] = {0, 0, -1};

          double hit_z = 0.0;
          if (!CastTerrainHitZ(model, data, origin, ray, hit_z))
          {
            // A miss is unknown terrain. For sim testing, fill it as flat
            // ground instead of emitting 0.0, which looks like a false obstacle.
            hit_z = 0.0;
          }

          double value = sensor_height - hit_z - offset;
          if (value_clip > 0.0)
          {
            value = std::clamp(value, -value_clip, value_clip);
          }
          values.push_back(static_cast<float>(value));
        }
      }
    }

    static bool CastTerrainHitZ(
      const mjModel* model,
      const mjData* data,
      const mjtNum ray_origin[3],
      const mjtNum ray[3],
      double& hit_z)
    {
      mjtNum origin[3] = {ray_origin[0], ray_origin[1], ray_origin[2]};

      for (int attempt = 0; attempt < 32; ++attempt)
      {
        int geom_id = -1;
        const mjtNum distance = mj_ray(model, data, origin, ray, nullptr, 1, -1, &geom_id);
        if (distance <= 0 || geom_id < 0)
        {
          return false;
        }

        const int body_id = model->geom_bodyid[geom_id];
        const double candidate_z = static_cast<double>(origin[2] + distance * ray[2]);

        // IsaacLab casts only against the terrain mesh. In these MuJoCo scenes,
        // terrain/floor/stones are worldbody geoms, while robot collision geoms
        // belong to non-world bodies. Skip robot hits and keep casting down.
        if (body_id == 0)
        {
          hit_z = candidate_z;
          return true;
        }

        constexpr mjtNum kSkipEpsilon = 0.02;
        origin[0] = origin[0] + ray[0] * (distance + kSkipEpsilon);
        origin[1] = origin[1] + ray[1] * (distance + kSkipEpsilon);
        origin[2] = origin[2] + ray[2] * (distance + kSkipEpsilon);
      }

      return false;
    }

    static void WriteHeightScanFile(const std::vector<float>& values)
    {
      const std::string& output = param::config.height_scan_bridge_file;
      if (output.empty() || values.empty())
      {
        return;
      }

      const std::string tmp = output + ".mujoco.tmp";
      std::ofstream file(tmp, std::ios::binary);
      if (!file)
      {
        return;
      }

      file << "UTHEIGHT1 " << values.size() << "\n";
      file.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
      file.close();

      std::rename(tmp.c_str(), output.c_str());
    }

    const mjModel* model_ = nullptr;
    int body_id_ = 0;
    double last_write_time_ = -1.0e9;
  };

  DepthBridge depth_bridge;
  HeightScanBridge height_scan_bridge;
  TouchdownTruthLogger touchdown_truth_logger;

  //---------------------------------------- plugin handling -----------------------------------------

  // return the path to the directory containing the current executable
  // used to determine the location of auto-loaded plugin libraries
  std::string getExecutableDir()
  {
#if defined(_WIN32) || defined(__CYGWIN__)
    constexpr char kPathSep = '\\';
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      DWORD buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
        if (written < buf_size)
        {
          success = true;
        }
        else if (written == buf_size)
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
        else
        {
          std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
          return "";
        }
      }
      return realpath.get();
    }();
#else
    constexpr char kPathSep = '/';
#if defined(__APPLE__)
    std::unique_ptr<char[]> buf(nullptr);
    {
      std::uint32_t buf_size = 0;
      _NSGetExecutablePath(nullptr, &buf_size);
      buf.reset(new char[buf_size]);
      if (!buf)
      {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }
      if (_NSGetExecutablePath(buf.get(), &buf_size))
      {
        std::cerr << "unexpected error from _NSGetExecutablePath\n";
      }
    }
    const char *path = buf.get();
#else
    const char *path = "/proc/self/exe";
#endif
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      std::uint32_t buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        std::size_t written = readlink(path, realpath.get(), buf_size);
        if (written < buf_size)
        {
          realpath.get()[written] = '\0';
          success = true;
        }
        else if (written == -1)
        {
          if (errno == EINVAL)
          {
            // path is already not a symlink, just use it
            return path;
          }

          std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
          return "";
        }
        else
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
      }
      return realpath.get();
    }();
#endif

    if (realpath.empty())
    {
      return "";
    }

    for (std::size_t i = realpath.size() - 1; i > 0; --i)
    {
      if (realpath.c_str()[i] == kPathSep)
      {
        return realpath.substr(0, i);
      }
    }

    // don't scan through the entire file system's root
    return "";
  }

  // scan for libraries in the plugin directory to load additional plugins
  void scanPluginLibraries()
  {
    // check and print plugins that are linked directly into the executable
    int nplugin = mjp_pluginCount();
    if (nplugin)
    {
      std::printf("Built-in plugins:\n");
      for (int i = 0; i < nplugin; ++i)
      {
        std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
      }
    }

    // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
    const std::string sep = "\\";
#else
    const std::string sep = "/";
#endif

    // try to open the ${EXECDIR}/plugin directory
    // ${EXECDIR} is the directory containing the simulate binary itself
    const std::string executable_dir = getExecutableDir();
    if (executable_dir.empty())
    {
      return;
    }

    const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
    mj_loadAllPluginLibraries(
        plugin_dir.c_str(), +[](const char *filename, int first, int count)
                            {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        } });
  }

  //------------------------------------------- simulation -------------------------------------------

  mjModel *LoadModel(const char *file, mj::Simulate &sim)
  {
    // this copy is needed so that the mju::strlen call below compiles
    char filename[mj::Simulate::kMaxFilenameLength];
    mju::strcpy_arr(filename, file);

    // make sure filename is not empty
    if (!filename[0])
    {
      return nullptr;
    }

    // load and compile
    char loadError[kErrorLength] = "";
    mjModel *mnew = 0;
    if (mju::strlen_arr(filename) > 4 &&
        !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                      mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4))
    {
      mnew = mj_loadModel(filename, nullptr);
      if (!mnew)
      {
        mju::strcpy_arr(loadError, "could not load binary model");
      }
    }
    else
    {
      mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);
      // remove trailing newline character from loadError
      if (loadError[0])
      {
        int error_length = mju::strlen_arr(loadError);
        if (loadError[error_length - 1] == '\n')
        {
          loadError[error_length - 1] = '\0';
        }
      }
    }

    mju::strcpy_arr(sim.load_error, loadError);

    if (!mnew)
    {
      std::printf("%s\n", loadError);
      return nullptr;
    }

    // compiler warning: print and pause
    if (loadError[0])
    {
      // mj_forward() below will print the warning message
      std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
      sim.run = 0;
    }

    return mnew;
  }

  // simulate in background thread (while rendering in main thread)
  void PhysicsLoop(mj::Simulate &sim)
  {
    // cpu-sim syncronization point
    std::chrono::time_point<mj::Simulate::Clock> syncCPU;
    mjtNum syncSim = 0;

    // ChannelFactory::Instance()->Init(0);
    // UnitreeDds ud(d);

    // run until asked to exit
    while (!sim.exitrequest.load())
    {
      if (sim.droploadrequest.load())
      {
        sim.LoadMessage(sim.dropfilename);
        mjModel *mnew = LoadModel(sim.dropfilename, sim);
        sim.droploadrequest.store(false);

        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.dropfilename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = (mjtNum *)malloc(sizeof(mjtNum) * m->nu);
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      if (sim.uiloadrequest.load())
      {
        sim.uiloadrequest.fetch_sub(1);
        sim.LoadMessage(sim.filename);
        mjModel *mnew = LoadModel(sim.filename, sim);
        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.filename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      // sleep for 1 ms or yield, to let main thread run
      //  yield results in busy wait - which has better timing but kills battery life
      if (sim.run && sim.busywait)
      {
        std::this_thread::yield();
      }
      else
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      {
        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        // run only if model is present
        if (m)
        {
          // running
          if (sim.run)
          {
            bool stepped = false;

            // record cpu time at start of iteration
            const auto startCPU = mj::Simulate::Clock::now();

            // elapsed CPU and simulation time since last sync
            const auto elapsedCPU = startCPU - syncCPU;
            double elapsedSim = d->time - syncSim;

            // inject noise
            if (sim.ctrl_noise_std)
            {
              // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
              mjtNum rate = mju_exp(-m->opt.timestep / mju_max(sim.ctrl_noise_rate, mjMINVAL));
              mjtNum scale = sim.ctrl_noise_std * mju_sqrt(1 - rate * rate);

              for (int i = 0; i < m->nu; i++)
              {
                // update noise
                ctrlnoise[i] = rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

                // apply noise
                d->ctrl[i] = ctrlnoise[i];
              }
            }

            // requested slow-down factor
            double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

            // misalignment condition: distance from target sim time is bigger than syncmisalign
            bool misaligned =
                mju_abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

            // out-of-sync (for any reason): reset sync times, step
            if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
                misaligned || sim.speed_changed)
            {
              // re-sync
              syncCPU = startCPU;
              syncSim = d->time;
              sim.speed_changed = false;

              // run single step, let next iteration deal with timing
              mj_step(m, d);
              touchdown_truth_logger.MaybeWrite(m, d);
              stepped = true;
            }

            // in-sync: step until ahead of cpu
            else
            {
              bool measured = false;
              mjtNum prevSim = d->time;

              double refreshTime = simRefreshFraction / sim.refresh_rate;

              // step while sim lags behind cpu and within refreshTime
              while (Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                     mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime))
              {
                // measure slowdown before first step
                if (!measured && elapsedSim)
                {
                  sim.measured_slowdown =
                      std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                  measured = true;
                }

                // elastic band on base link
                if (param::config.enable_elastic_band == 1)
                {
                  const int wrench = param::config.band_attached_link;
                  const int body_id = wrench / 6;
                  if (wrench >= 0 && wrench + 5 < 6 * m->nbody)
                  {
                    for (int i = 0; i < 6; ++i)
                    {
                      d->xfrc_applied[wrench + i] = 0;
                    }
                  }
                  if (elastic_band.enable_)
                  {
                    std::vector<double> x = {d->qpos[0], d->qpos[1], d->qpos[2]};
                    std::vector<double> dx = {d->qvel[0], d->qvel[1], d->qvel[2]};

                    elastic_band.Advance(x, dx);

                    if (wrench >= 0 && wrench + 5 < 6 * m->nbody)
                    {
                      d->xfrc_applied[wrench] = elastic_band.f_[0];
                      d->xfrc_applied[wrench + 1] = elastic_band.f_[1];
                      d->xfrc_applied[wrench + 2] = elastic_band.f_[2];
                      ApplyElasticBandAutoAlign(m, d, body_id);
                    }
                  }
                }

                // call mj_step
                mj_step(m, d);
                touchdown_truth_logger.MaybeWrite(m, d);
                stepped = true;

                // break if reset
                if (d->time < prevSim)
                {
                  break;
                }
              }
            }

            // save current state to history buffer
            if (stepped)
            {
              sim.AddToHistory();
              depth_bridge.MaybeWrite(m, d);
              height_scan_bridge.MaybeWrite(m, d);
            }
          }

          // paused
          else
          {
            // run mj_forward, to update rendering and joint sliders
            mj_forward(m, d);
            depth_bridge.MaybeWrite(m, d);
            height_scan_bridge.MaybeWrite(m, d);
            sim.speed_changed = true;
          }
        }
      } // release std::lock_guard<std::mutex>
    }
  }
} // namespace

//-------------------------------------- physics_thread --------------------------------------------

void PhysicsThread(mj::Simulate *sim, const char *filename)
{
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr)
  {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
    if (m)
      d = mj_makeData(m);
    if (d)
    {
      sim->Load(m, d, filename);
      mj_forward(m, d);

      // allocate ctrlnoise
      free(ctrlnoise);
      ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
      mju_zero(ctrlnoise, m->nu);
    }
    else
    {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  free(ctrlnoise);
  mj_deleteData(d);
  mj_deleteModel(m);

  exit(0);
}

void *UnitreeSdk2BridgeThread(void *arg)
{
  // Wait for mujoco data
  while (true)
  {
    if (d)
    {
      std::cout << "Mujoco data is prepared" << std::endl;
      break;
    }
    usleep(500000);
  }

  unitree::robot::ChannelFactory::Instance()->Init(param::config.domain_id, param::config.interface);


  int body_id = mj_name2id(m, mjOBJ_BODY, "torso_link");
  if (body_id < 0) {
    body_id = mj_name2id(m, mjOBJ_BODY, "base_link");
  }
  if (body_id < 0) {
    body_id = mj_name2id(m, mjOBJ_BODY, "pelvis");
  }
  if (body_id < 0) {
    body_id = 0;
  }
  param::config.band_attached_link = 6 * body_id;
  
  std::unique_ptr<UnitreeSDK2BridgeBase> interface = nullptr;
  if (m->nu > NUM_MOTOR_IDL_GO) {
    interface = std::make_unique<G1Bridge>(m, d);
  } else {
    interface = std::make_unique<Go2Bridge>(m, d);
  }
  interface->start();
  
  while (true)
  {
    sleep(1);
  }
}
//------------------------------------------ main --------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char *title, const char *msg);
static const char *rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char *msg)
{
  rosetta_error_msg = msg;
}
#endif

// user keyboard callback
void user_key_cb(GLFWwindow* window, int key, int scancode, int act, int mods) {
  if (act==GLFW_PRESS)
  {
    if(param::config.enable_elastic_band == 1) {
      if (key==GLFW_KEY_9) {
        elastic_band.enable_ = !elastic_band.enable_;
      } else if (key==GLFW_KEY_7 || key==GLFW_KEY_UP) {
        elastic_band.length_ -= 0.1;
      } else if (key==GLFW_KEY_8 || key==GLFW_KEY_DOWN) {
        elastic_band.length_ += 0.1;
      }
    }
    if(key==GLFW_KEY_BACKSPACE) {
      mj_resetData(m, d);
      mj_forward(m, d);
    }
  }
}

// run event loop
int main(int argc, char **argv)
{

  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg)
  {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version())
  {
    mju_error("Headers and library have different versions");
  }

  // scan for libraries in the plugin directory to load additional plugins
  scanPluginLibraries();

  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // Load simulation configuration
  std::filesystem::path proj_dir = std::filesystem::path(getExecutableDir()).parent_path();
  param::config.load_from_yaml(proj_dir / "config.yaml");
  param::helper(argc, argv);
  if(param::config.robot_scene.is_relative()) {
    param::config.robot_scene = proj_dir.parent_path() / "unitree_robots" / param::config.robot / param::config.robot_scene;
  }

  // simulate object encapsulates the UI
  auto sim = std::make_unique<mj::Simulate>(
    std::make_unique<mj::GlfwAdapter>(),
    &cam, &opt, &pert, /* is_passive = */ false);

  std::thread unitree_thread(UnitreeSdk2BridgeThread, nullptr);

  // start physics thread
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), param::config.robot_scene.c_str());
  // start simulation UI loop (blocking call)
  glfwSetKeyCallback(static_cast<mj::GlfwAdapter*>(sim->platform_ui.get())->window_,user_key_cb);
  sim->RenderLoop();
  physicsthreadhandle.join();

  pthread_exit(NULL);
  return 0;
}
