#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace
{

struct VectorStats
{
    float min = 0.0f;
    float max = 0.0f;
    float mean = 0.0f;
};

VectorStats stats_for(const std::vector<float>& values)
{
    VectorStats stats;
    if (values.empty()) {
        return stats;
    }

    const auto minmax = std::minmax_element(values.begin(), values.end());
    stats.min = *minmax.first;
    stats.max = *minmax.second;
    stats.mean = std::accumulate(values.begin(), values.end(), 0.0f) /
        static_cast<float>(values.size());
    return stats;
}

std::string head_values(const std::vector<float>& values, const size_t count)
{
    std::ostringstream out;
    out << "[";
    const size_t n = std::min(values.size(), count);
    for (size_t i = 0; i < n; ++i) {
        if (i != 0) {
            out << ",";
        }
        out << std::fixed << std::setprecision(3) << values[i];
    }
    if (values.size() > n) {
        out << ",...";
    }
    out << "]";
    return out.str();
}

std::string leg_pair_values(const std::vector<float>& values)
{
    const std::array<std::pair<size_t, size_t>, 6> pairs = {{
        {0, 1},    // hip pitch
        {3, 4},    // hip roll
        {7, 8},    // hip yaw
        {11, 12},  // knee
        {15, 16},  // ankle pitch
        {19, 20},  // ankle roll
    }};
    const std::array<const char*, 6> names = {{
        "hp", "hr", "hy", "kn", "ap", "ar"
    }};

    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        const auto [left, right] = pairs[i];
        out << names[i] << "=(";
        if (left < values.size() && right < values.size()) {
            out << std::fixed << std::setprecision(3)
                << values[left] << "," << values[right];
        } else {
            out << "nan,nan";
        }
        out << ")";
    }
    out << "]";
    return out.str();
}

}

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        // TODO: smooth and limit the velocity commands
        cmd = key_commands[key];
    }
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");
    configure_diagnostics();

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    if (run_entry_posture()) {
        return;
    }

    auto action = policy_faulted_ || use_stand_hold_target()
        ? stand_hold_target()
        : env->action_manager->processed_actions();
    const auto& joint_ids = env->robot->data.joint_ids_map;
    const int count = std::min(action.size(), joint_ids.size());
    for(int i(0); i < count; i++) {
        lowcmd->msg_.motor_cmd()[static_cast<int>(joint_ids[i])].q() = action[i];
    }
    apply_locked_joint_commands(false);
}

bool State_RLBase::use_stand_hold_target() const
{
    const auto cfg = env->cfg["stand_hold"];
    if (!cfg || !cfg["use_on_zero_command"].as<bool>(false)) {
        return false;
    }
    return env->hold_default_on_zero_command();
}

std::vector<float> State_RLBase::stand_hold_target() const
{
    const auto active_count = env->robot->data.joint_ids_map.size();
    std::vector<float> target(
        env->robot->data.default_joint_pos.data(),
        env->robot->data.default_joint_pos.data() +
            env->robot->data.default_joint_pos.size());

    const auto cfg = env->cfg["stand_hold"];
    if (!cfg || !cfg["positions"]) {
        return target;
    }

    try {
        const auto positions = cfg["positions"].as<std::vector<float>>();
        if (positions.size() == active_count) {
            return positions;
        }
    } catch (const std::exception&) {
    }
    return target;
}

void State_RLBase::configure_diagnostics()
{
    const auto cfg = env->cfg["diagnostics"];
    if (!cfg) {
        return;
    }

    diagnostics_enabled_ = cfg["enabled"].as<bool>(false);
    diagnostics_interval_steps_ =
        std::max<long>(1, cfg["log_interval_steps"].as<long>(25));
    diagnostics_warmup_steps_ =
        std::max<long>(0, cfg["warmup_steps"].as<long>(10));

    if (diagnostics_enabled_) {
        spdlog::info(
            "[{}] diagnostics enabled interval_steps={} warmup_steps={}",
            getStateString(),
            diagnostics_interval_steps_,
            diagnostics_warmup_steps_);
    }
}

void State_RLBase::log_policy_diagnostics()
{
    if (!diagnostics_enabled_) {
        return;
    }

    const long step = env->episode_length;
    if (step > diagnostics_warmup_steps_ &&
        step % diagnostics_interval_steps_ != 0) {
        return;
    }

    const auto command = isaaclab::mdp::velocity_commands(env.get(), YAML::Node());
    std::vector<float> height;
    if (env->cfg["observations"]["height_scan"]) {
        height = isaaclab::mdp::height_scan(
            env.get(),
            env->cfg["observations"]["height_scan"]["params"]);
    }
    const auto raw_action = env->action_manager->action();
    const auto q_target = use_stand_hold_target()
        ? stand_hold_target()
        : env->action_manager->processed_actions();

    std::vector<float> q_rel(static_cast<size_t>(env->robot->data.joint_pos.size()), 0.0f);
    std::vector<float> q_vel(static_cast<size_t>(env->robot->data.joint_vel.size()), 0.0f);
    for (size_t i = 0; i < q_rel.size(); ++i) {
        q_rel[i] = env->robot->data.joint_pos[i] -
            env->robot->data.default_joint_pos[static_cast<int>(i)];
        q_vel[i] = env->robot->data.joint_vel[i];
    }

    const auto height_stats = stats_for(height);
    const auto raw_stats = stats_for(raw_action);
    const auto q_target_stats = stats_for(q_target);
    const auto q_rel_stats = stats_for(q_rel);
    const auto q_vel_stats = stats_for(q_vel);

    const auto& gravity = env->robot->data.projected_gravity_b;
    const auto& gyro = env->robot->data.root_ang_vel_b;
    auto* joystick = env->robot->data.joystick;
    spdlog::info(
        "[{}] step={} stick=[{:.3f},{:.3f},{:.3f}] cmd={} height[min,max,mean]=[{:.3f},{:.3f},{:.3f}] "
        "raw[min,max,mean]=[{:.3f},{:.3f},{:.3f}] q_target[min,max]=[{:.3f},{:.3f}] "
        "q_rel[min,max]=[{:.3f},{:.3f}] q_vel[min,max]=[{:.3f},{:.3f}] "
        "gravity=[{:.3f},{:.3f},{:.3f}] "
        "gyro=[{:.3f},{:.3f},{:.3f}] raw0={} qtarget0={} qrel0={} qvel0={} "
        "raw_leg={} qtarget_leg={} qrel_leg={} qvel_leg={}",
        getStateString(),
        step,
        joystick ? joystick->ly() : 0.0f,
        joystick ? joystick->lx() : 0.0f,
        joystick ? joystick->rx() : 0.0f,
        head_values(command, 3),
        height_stats.min,
        height_stats.max,
        height_stats.mean,
        raw_stats.min,
        raw_stats.max,
        raw_stats.mean,
        q_target_stats.min,
        q_target_stats.max,
        q_rel_stats.min,
        q_rel_stats.max,
        q_vel_stats.min,
        q_vel_stats.max,
        gravity[0],
        gravity[1],
        gravity[2],
        gyro[0],
        gyro[1],
        gyro[2],
        head_values(raw_action, 6),
        head_values(q_target, 6),
        head_values(q_rel, 6),
        head_values(q_vel, 6),
        leg_pair_values(raw_action),
        leg_pair_values(q_target),
        leg_pair_values(q_rel),
        leg_pair_values(q_vel));
}
