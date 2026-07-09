#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"

#include <string>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // Unitree DDS Config
    //
    // Real robot:
    //   ./g1_ctrl --network=eth0
    //   domain_id = 0, interface = eth0
    //
    // MuJoCo simulation:
    //   ./g1_ctrl --network=sim
    //   domain_id = 1, interface = lo
    //
    const std::string network_arg = vm["network"].as<std::string>();

    if (network_arg == "sim" || network_arg == "mujoco" || network_arg == "unitree_mujoco")
    {
        spdlog::info("Running in MuJoCo simulation mode.");
        spdlog::info("DDS domain_id = 1, interface = lo");
        unitree::robot::ChannelFactory::Instance()->Init(1, "lo");
    }
    else
    {
        spdlog::info("Running in real robot mode.");
        spdlog::info("DDS domain_id = 0, interface = {}", network_arg);
        unitree::robot::ChannelFactory::Instance()->Init(0, network_arg);
    }

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R2 + A] to start controlling the robot.\n";
    std::cout << "And then press [R1 + A/B/Y/X] to control the robot dance.\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}
