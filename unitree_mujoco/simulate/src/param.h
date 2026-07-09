#pragma once

#include <iostream>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace param
{

inline struct SimulationConfig
{
    std::string robot;
    std::filesystem::path robot_scene;

    int domain_id;
    std::string interface;

    int use_joystick;
    std::string joystick_type;
    std::string joystick_device;
    int joystick_bits;

    int print_scene_information;

    int enable_elastic_band;
    int band_attached_link = 0;

    int enable_depth_bridge = 1;
    std::string depth_bridge_file = "/tmp/unitree_g1_front_depth.bin";
    std::string depth_camera_body = "torso_link";
    int depth_width = 64;
    int depth_height = 36;
    double depth_min = 0.01;
    double depth_cutoff = 3.0;
    double depth_fovy = 55.2;
    double depth_fps = 50.0;
    double depth_pitch_deg = -47.6;
    double depth_pos_x = 0.10;
    double depth_pos_y = 0.0;
    double depth_pos_z = 0.45;

    int enable_height_scan_bridge = 1;
    std::string height_scan_bridge_file = "/tmp/unitree_g1_height_scan.bin";
    std::string height_scan_body = "pelvis";
    double height_scan_size_x = 1.0;
    double height_scan_size_y = 1.0;
    double height_scan_resolution = 0.1;
    double height_scan_ray_start_z = 20.0;
    double height_scan_offset = 0.5;
    double height_scan_fps = 50.0;

    int enable_touchdown_truth_log = 1;
    std::string touchdown_truth_log_file =
        "/tmp/mjlab_g1_touchdown_truth.csv";
    int touchdown_truth_use_latest_run_dir = 0;
    std::string touchdown_truth_latest_run_dir_file =
        "/tmp/mjlab_g1_latest_run_dir.txt";
    int touchdown_truth_timestamp_run_dir = 1;
    int touchdown_truth_flush_interval_events = 10;
    int touchdown_truth_state_interval_ms = 20;
    int touchdown_truth_min_air_ms = 50;
    int touchdown_truth_cooldown_ms = 120;

    void load_from_yaml(const std::string &filename)
    {
        auto cfg = YAML::LoadFile(filename);
        try
        {
            robot = cfg["robot"].as<std::string>();
            robot_scene = cfg["robot_scene"].as<std::string>();
            domain_id = cfg["domain_id"].as<int>();
            interface = cfg["interface"].as<std::string>();
            use_joystick = cfg["use_joystick"].as<int>();
            joystick_type = cfg["joystick_type"].as<std::string>();
            joystick_device = cfg["joystick_device"].as<std::string>();
            joystick_bits = cfg["joystick_bits"].as<int>();
            print_scene_information = cfg["print_scene_information"].as<int>();
            enable_elastic_band = cfg["enable_elastic_band"].as<int>();

            if (cfg["enable_depth_bridge"]) enable_depth_bridge = cfg["enable_depth_bridge"].as<int>();
            if (cfg["depth_bridge_file"]) depth_bridge_file = cfg["depth_bridge_file"].as<std::string>();
            if (cfg["depth_camera_body"]) depth_camera_body = cfg["depth_camera_body"].as<std::string>();
            if (cfg["depth_width"]) depth_width = cfg["depth_width"].as<int>();
            if (cfg["depth_height"]) depth_height = cfg["depth_height"].as<int>();
            if (cfg["depth_min"]) depth_min = cfg["depth_min"].as<double>();
            if (cfg["depth_cutoff"]) depth_cutoff = cfg["depth_cutoff"].as<double>();
            if (cfg["depth_fovy"]) depth_fovy = cfg["depth_fovy"].as<double>();
            if (cfg["depth_fps"]) depth_fps = cfg["depth_fps"].as<double>();
            if (cfg["depth_pitch_deg"]) depth_pitch_deg = cfg["depth_pitch_deg"].as<double>();
            if (cfg["depth_pos_x"]) depth_pos_x = cfg["depth_pos_x"].as<double>();
            if (cfg["depth_pos_y"]) depth_pos_y = cfg["depth_pos_y"].as<double>();
            if (cfg["depth_pos_z"]) depth_pos_z = cfg["depth_pos_z"].as<double>();
            if (cfg["enable_height_scan_bridge"]) enable_height_scan_bridge = cfg["enable_height_scan_bridge"].as<int>();
            if (cfg["height_scan_bridge_file"]) height_scan_bridge_file = cfg["height_scan_bridge_file"].as<std::string>();
            if (cfg["height_scan_body"]) height_scan_body = cfg["height_scan_body"].as<std::string>();
            if (cfg["height_scan_size_x"]) height_scan_size_x = cfg["height_scan_size_x"].as<double>();
            if (cfg["height_scan_size_y"]) height_scan_size_y = cfg["height_scan_size_y"].as<double>();
            if (cfg["height_scan_resolution"]) height_scan_resolution = cfg["height_scan_resolution"].as<double>();
            if (cfg["height_scan_ray_start_z"]) height_scan_ray_start_z = cfg["height_scan_ray_start_z"].as<double>();
            if (cfg["height_scan_offset"]) height_scan_offset = cfg["height_scan_offset"].as<double>();
            if (cfg["height_scan_fps"]) height_scan_fps = cfg["height_scan_fps"].as<double>();
            if (cfg["enable_touchdown_truth_log"]) enable_touchdown_truth_log = cfg["enable_touchdown_truth_log"].as<int>();
            if (cfg["touchdown_truth_log_file"]) touchdown_truth_log_file = cfg["touchdown_truth_log_file"].as<std::string>();
            if (cfg["touchdown_truth_use_latest_run_dir"]) touchdown_truth_use_latest_run_dir = cfg["touchdown_truth_use_latest_run_dir"].as<int>();
            if (cfg["touchdown_truth_latest_run_dir_file"]) touchdown_truth_latest_run_dir_file = cfg["touchdown_truth_latest_run_dir_file"].as<std::string>();
            if (cfg["touchdown_truth_timestamp_run_dir"]) touchdown_truth_timestamp_run_dir = cfg["touchdown_truth_timestamp_run_dir"].as<int>();
            if (cfg["touchdown_truth_flush_interval_events"]) touchdown_truth_flush_interval_events = cfg["touchdown_truth_flush_interval_events"].as<int>();
            if (cfg["touchdown_truth_state_interval_ms"]) touchdown_truth_state_interval_ms = cfg["touchdown_truth_state_interval_ms"].as<int>();
            if (cfg["touchdown_truth_min_air_ms"]) touchdown_truth_min_air_ms = cfg["touchdown_truth_min_air_ms"].as<int>();
            if (cfg["touchdown_truth_cooldown_ms"]) touchdown_truth_cooldown_ms = cfg["touchdown_truth_cooldown_ms"].as<int>();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }
} config;

/* ---------- Command Line Parameters ---------- */
namespace po = boost::program_options;

//※ This function must be called at the beginning of main() function
inline po::variables_map helper(int argc, char** argv)
{
    po::options_description desc("Unitree Mujoco");
    desc.add_options()
        ("help,h", "Show help message")
        ("domain_id,i", po::value<int>(&config.domain_id), "DDS domain ID; -i 0")
        ("network,n", po::value<std::string>(&config.interface), "DDS network interface; -n eth0")
        ("robot,r", po::value<std::string>(&config.robot), "Robot type; -r go2")
        ("scene,s", po::value<std::filesystem::path>(&config.robot_scene), "Robot scene file; -s scene_terrain.xml")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(0);
    }

    return vm;
}

}
