// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/manager/action_manager.h"

#include <algorithm>

namespace isaaclab
{

class JointAction : public ActionTerm
{
public:
    JointAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :ActionTerm(cfg, env)
    {
        if(cfg["joint_ids"].IsNull()) {
            _action_dim = env->robot->data.joint_ids_map.size();
        } else {
            _joint_ids = cfg["joint_ids"].as<std::vector<int>>();
            _action_dim = _joint_ids.size();
        }
        _raw_actions.resize(_action_dim, 0.0f);
        _processed_actions.resize(_action_dim, 0.0f);
        if(!cfg["scale"].IsNull()) {
            _scale = cfg["scale"].as<std::vector<float>>();
        }
        if(!cfg["offset"].IsNull()) {
            _offset = cfg["offset"].as<std::vector<float>>();
        }
        if(!cfg["clip"].IsNull()) {
            _clip = cfg["clip"].as<std::vector<std::vector<float> >>();
        }
        if(cfg["raw_clip"] && !cfg["raw_clip"].IsNull()) {
            const auto raw_clip = cfg["raw_clip"];
            if(raw_clip.IsSequence() && raw_clip.size() == 2 &&
               raw_clip[0].IsScalar() && raw_clip[1].IsScalar()) {
                _raw_clip.push_back({raw_clip[0].as<float>(), raw_clip[1].as<float>()});
            } else {
                _raw_clip = raw_clip.as<std::vector<std::vector<float> >>();
            }
        }
        if(cfg["raw_filter_alpha"] && !cfg["raw_filter_alpha"].IsNull()) {
            _raw_filter_alpha = std::clamp(cfg["raw_filter_alpha"].as<float>(), 0.0f, 0.99f);
        }
    }

    virtual void process_actions(std::vector<float> actions)
    {
        // TODO: modify action by joint_ids
        _raw_actions = actions;
        if(!_raw_clip.empty()) {
            for(int i(0); i<_action_dim; ++i) {
                const auto& limits = _raw_clip.size() == 1 ? _raw_clip[0] : _raw_clip[i];
                if(limits.size() >= 2) {
                    _raw_actions[i] = std::clamp(_raw_actions[i], limits[0], limits[1]);
                }
            }
        }
        if(_raw_filter_alpha > 0.0f) {
            if(!_raw_filter_initialized || _prev_raw_actions.size() != _raw_actions.size()) {
                _prev_raw_actions = _raw_actions;
                _raw_filter_initialized = true;
            } else {
                for(int i(0); i<_action_dim; ++i) {
                    _raw_actions[i] = _raw_filter_alpha * _prev_raw_actions[i]
                        + (1.0f - _raw_filter_alpha) * _raw_actions[i];
                }
                _prev_raw_actions = _raw_actions;
            }
        }
        for(int i(0); i<_action_dim; ++i)
        {
            if(!_scale.empty()) {
                _processed_actions[i] = _raw_actions[i] * _scale[i];
            } else {
                _processed_actions[i] = _raw_actions[i];
            }
            if(!_offset.empty()) {
                _processed_actions[i] += _offset[i];
            }
        }
        if(!_clip.empty())
        {
            for(int i(0); i<_action_dim; ++i) {
                _processed_actions[i] = std::clamp(_processed_actions[i], _clip[i][0], _clip[i][1]);
            }
        }
    }


    int action_dim() 
    {
        return _action_dim;
    }

    std::vector<float> raw_actions() 
    {
        return _raw_actions;
    }
    
    std::vector<float> processed_actions() 
    {
        return _processed_actions;
    }

    void reset()
    {
        _raw_actions.assign(_action_dim, 0.0f);
        _prev_raw_actions.assign(_action_dim, 0.0f);
        _raw_filter_initialized = false;
        process_actions(_raw_actions);
    }

protected:
    int _action_dim;
    std::vector<int> _joint_ids;

    std::vector<float> _raw_actions;
    std::vector<float> _prev_raw_actions;
    std::vector<float> _processed_actions;

    std::vector<float> _scale;
    std::vector<float> _offset;
    std::vector<std::vector<float> > _raw_clip;
    float _raw_filter_alpha = 0.0f;
    bool _raw_filter_initialized = false;
    std::vector<std::vector<float> > _clip;
};


class JointPositionAction : public JointAction
{
public:
    JointPositionAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

class JointVelocityAction : public JointAction
{
public:
    JointVelocityAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

REGISTER_ACTION(JointPositionAction);
REGISTER_ACTION(JointVelocityAction);

};
