#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

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
    auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
    auto robot = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        robot
    );
    env->reset_callback = [this]() { reset_foot_event_runtime(); };

    if (deploy_cfg["camera"] && deploy_cfg["camera"]["front_depth"]) {
        const auto camera_cfg = deploy_cfg["camera"]["front_depth"];
        const int width = camera_cfg["width"].as<int>(64);
        const int height = camera_cfg["height"].as<int>(36);
        const float min_depth = camera_cfg["min_depth"].as<float>(0.01f);
        const float cutoff_distance = camera_cfg["cutoff_distance"].as<float>(3.0f);
        const bool normalize = camera_cfg["normalize"].as<bool>(true);
        const int rotate_k = camera_cfg["rotate_k"].as<int>(0);
        const int stack_length = camera_cfg["stack_length"].as<int>(
            camera_cfg["channels"].as<int>(8)
        );
        const std::string topic = camera_cfg["topic"].as<std::string>("/camera/depth/image_rect_raw");
        const std::string bridge_file = camera_cfg["bridge_file"].as<std::string>("/tmp/unitree_g1_front_depth.bin");

        robot->configure_depth_camera(
            "front_depth",
            topic,
            width,
            height,
            min_depth,
            cutoff_distance,
            normalize,
            rotate_k,
            stack_length,
            bridge_file
        );
    }

    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

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

    const auto action = env->action_manager->processed_actions();
    const auto& joint_ids = env->robot->data.joint_ids_map;
    const int count = std::min(action.size(), joint_ids.size());
    for (int i = 0; i < count; ++i) {
        lowcmd->msg_.motor_cmd()[joint_ids[i]].q() = action[i];
    }
    apply_locked_joint_commands(false);
}
