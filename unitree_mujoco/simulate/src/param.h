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
    int elastic_band_auto_align = 1;
    double elastic_band_target_yaw_deg = 0.0;
    double elastic_band_align_kp_upright = 140.0;
    double elastic_band_align_kd_upright = 18.0;
    double elastic_band_align_kp_yaw = 80.0;
    double elastic_band_align_kd_yaw = 12.0;
    double elastic_band_align_torque_limit = 120.0;

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
    std::string height_scan_mode = "ideal";
    std::string height_scan_body = "pelvis";
    double height_scan_size_x = 1.0;
    double height_scan_size_y = 1.0;
    double height_scan_resolution = 0.1;
    double height_scan_ray_start_z = 20.0;
    double height_scan_offset = 0.5;
    double height_scan_fps = 50.0;
    int height_scan_sensor_height_includes_ray_start_z = 0;
    double height_scan_value_clip = 2.0;
    double height_scan_local_x_offset = 0.0;

    double height_scan_estimator_visible_x_min = 0.12;
    double height_scan_estimator_visible_x_max = 1.10;
    double height_scan_estimator_visible_y_min = -0.55;
    double height_scan_estimator_visible_y_max = 0.55;
    double height_scan_estimator_blind_x_min = -0.30;
    double height_scan_estimator_blind_x_max = 0.35;
    double height_scan_estimator_blind_y_min = -0.32;
    double height_scan_estimator_blind_y_max = 0.32;
    double height_scan_estimator_sample_resolution = 0.035;
    double height_scan_estimator_memory_resolution = 0.025;
    double height_scan_estimator_memory_max_age = 2.0;
    double height_scan_estimator_memory_max_distance = 2.5;
    double height_scan_estimator_query_radius = 0.075;
    double height_scan_estimator_query_percentile = 90.0;
    double height_scan_estimator_nominal_pelvis_height = 0.793;
    double height_scan_estimator_support_x_min = -0.24;
    double height_scan_estimator_support_x_max = 0.34;
    double height_scan_estimator_support_y_min = -0.28;
    double height_scan_estimator_support_y_max = 0.28;
    double height_scan_estimator_support_percentile = 82.0;
    double height_scan_estimator_base_z_rise_rate = 1.10;
    double height_scan_estimator_base_z_fall_rate = 0.45;
    double height_scan_estimator_fill_value = 0.28;
    int height_scan_estimator_use_last_for_unknown = 1;
    int height_scan_estimator_use_true_xy_yaw = 1;
    int height_scan_debug_write = 1;
    std::string height_scan_debug_csv_file = "/tmp/unitree_g1_height_scan_estimated.csv";
    std::string height_scan_debug_ppm_file = "/tmp/unitree_g1_height_scan_estimated.ppm";

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
            if (cfg["elastic_band_auto_align"]) elastic_band_auto_align = cfg["elastic_band_auto_align"].as<int>();
            if (cfg["elastic_band_target_yaw_deg"]) elastic_band_target_yaw_deg = cfg["elastic_band_target_yaw_deg"].as<double>();
            if (cfg["elastic_band_align_kp_upright"]) elastic_band_align_kp_upright = cfg["elastic_band_align_kp_upright"].as<double>();
            if (cfg["elastic_band_align_kd_upright"]) elastic_band_align_kd_upright = cfg["elastic_band_align_kd_upright"].as<double>();
            if (cfg["elastic_band_align_kp_yaw"]) elastic_band_align_kp_yaw = cfg["elastic_band_align_kp_yaw"].as<double>();
            if (cfg["elastic_band_align_kd_yaw"]) elastic_band_align_kd_yaw = cfg["elastic_band_align_kd_yaw"].as<double>();
            if (cfg["elastic_band_align_torque_limit"]) elastic_band_align_torque_limit = cfg["elastic_band_align_torque_limit"].as<double>();

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
            if (cfg["height_scan_mode"]) height_scan_mode = cfg["height_scan_mode"].as<std::string>();
            if (cfg["height_scan_body"]) height_scan_body = cfg["height_scan_body"].as<std::string>();
            if (cfg["height_scan_size_x"]) height_scan_size_x = cfg["height_scan_size_x"].as<double>();
            if (cfg["height_scan_size_y"]) height_scan_size_y = cfg["height_scan_size_y"].as<double>();
            if (cfg["height_scan_resolution"]) height_scan_resolution = cfg["height_scan_resolution"].as<double>();
            if (cfg["height_scan_ray_start_z"]) height_scan_ray_start_z = cfg["height_scan_ray_start_z"].as<double>();
            if (cfg["height_scan_offset"]) height_scan_offset = cfg["height_scan_offset"].as<double>();
            if (cfg["height_scan_fps"]) height_scan_fps = cfg["height_scan_fps"].as<double>();
            if (cfg["height_scan_sensor_height_includes_ray_start_z"]) {
                height_scan_sensor_height_includes_ray_start_z =
                    cfg["height_scan_sensor_height_includes_ray_start_z"].as<int>();
            }
            if (cfg["height_scan_value_clip"]) height_scan_value_clip = cfg["height_scan_value_clip"].as<double>();
            if (cfg["height_scan_local_x_offset"]) height_scan_local_x_offset = cfg["height_scan_local_x_offset"].as<double>();
            if (cfg["height_scan_estimator_visible_x_min"]) height_scan_estimator_visible_x_min = cfg["height_scan_estimator_visible_x_min"].as<double>();
            if (cfg["height_scan_estimator_visible_x_max"]) height_scan_estimator_visible_x_max = cfg["height_scan_estimator_visible_x_max"].as<double>();
            if (cfg["height_scan_estimator_visible_y_min"]) height_scan_estimator_visible_y_min = cfg["height_scan_estimator_visible_y_min"].as<double>();
            if (cfg["height_scan_estimator_visible_y_max"]) height_scan_estimator_visible_y_max = cfg["height_scan_estimator_visible_y_max"].as<double>();
            if (cfg["height_scan_estimator_blind_x_min"]) height_scan_estimator_blind_x_min = cfg["height_scan_estimator_blind_x_min"].as<double>();
            if (cfg["height_scan_estimator_blind_x_max"]) height_scan_estimator_blind_x_max = cfg["height_scan_estimator_blind_x_max"].as<double>();
            if (cfg["height_scan_estimator_blind_y_min"]) height_scan_estimator_blind_y_min = cfg["height_scan_estimator_blind_y_min"].as<double>();
            if (cfg["height_scan_estimator_blind_y_max"]) height_scan_estimator_blind_y_max = cfg["height_scan_estimator_blind_y_max"].as<double>();
            if (cfg["height_scan_estimator_sample_resolution"]) height_scan_estimator_sample_resolution = cfg["height_scan_estimator_sample_resolution"].as<double>();
            if (cfg["height_scan_estimator_memory_resolution"]) height_scan_estimator_memory_resolution = cfg["height_scan_estimator_memory_resolution"].as<double>();
            if (cfg["height_scan_estimator_memory_max_age"]) height_scan_estimator_memory_max_age = cfg["height_scan_estimator_memory_max_age"].as<double>();
            if (cfg["height_scan_estimator_memory_max_distance"]) height_scan_estimator_memory_max_distance = cfg["height_scan_estimator_memory_max_distance"].as<double>();
            if (cfg["height_scan_estimator_query_radius"]) height_scan_estimator_query_radius = cfg["height_scan_estimator_query_radius"].as<double>();
            if (cfg["height_scan_estimator_query_percentile"]) height_scan_estimator_query_percentile = cfg["height_scan_estimator_query_percentile"].as<double>();
            if (cfg["height_scan_estimator_nominal_pelvis_height"]) height_scan_estimator_nominal_pelvis_height = cfg["height_scan_estimator_nominal_pelvis_height"].as<double>();
            if (cfg["height_scan_estimator_support_x_min"]) height_scan_estimator_support_x_min = cfg["height_scan_estimator_support_x_min"].as<double>();
            if (cfg["height_scan_estimator_support_x_max"]) height_scan_estimator_support_x_max = cfg["height_scan_estimator_support_x_max"].as<double>();
            if (cfg["height_scan_estimator_support_y_min"]) height_scan_estimator_support_y_min = cfg["height_scan_estimator_support_y_min"].as<double>();
            if (cfg["height_scan_estimator_support_y_max"]) height_scan_estimator_support_y_max = cfg["height_scan_estimator_support_y_max"].as<double>();
            if (cfg["height_scan_estimator_support_percentile"]) height_scan_estimator_support_percentile = cfg["height_scan_estimator_support_percentile"].as<double>();
            if (cfg["height_scan_estimator_base_z_rise_rate"]) height_scan_estimator_base_z_rise_rate = cfg["height_scan_estimator_base_z_rise_rate"].as<double>();
            if (cfg["height_scan_estimator_base_z_fall_rate"]) height_scan_estimator_base_z_fall_rate = cfg["height_scan_estimator_base_z_fall_rate"].as<double>();
            if (cfg["height_scan_estimator_fill_value"]) height_scan_estimator_fill_value = cfg["height_scan_estimator_fill_value"].as<double>();
            if (cfg["height_scan_estimator_use_last_for_unknown"]) height_scan_estimator_use_last_for_unknown = cfg["height_scan_estimator_use_last_for_unknown"].as<int>();
            if (cfg["height_scan_estimator_use_true_xy_yaw"]) height_scan_estimator_use_true_xy_yaw = cfg["height_scan_estimator_use_true_xy_yaw"].as<int>();
            if (cfg["height_scan_debug_write"]) height_scan_debug_write = cfg["height_scan_debug_write"].as<int>();
            if (cfg["height_scan_debug_csv_file"]) height_scan_debug_csv_file = cfg["height_scan_debug_csv_file"].as<std::string>();
            if (cfg["height_scan_debug_ppm_file"]) height_scan_debug_ppm_file = cfg["height_scan_debug_ppm_file"].as<std::string>();
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
