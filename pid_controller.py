import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

import rclpy
# from stable_baselines3 import SAC
# from stable_baselines3.common.callbacks import (
#     BaseCallback,
#     CheckpointCallback,
#     EvalCallback
# )
# from stable_baselines3.common.monitor import Monitor
import tb4_drl_navigation.envs  # noqa: F401
import torch
import torch.nn as nn
import time
import yaml


# @dataclass(frozen=True)
# class EnvConfig:
#     env_id: str = 'Turtlebot4Env-v0'
#     world_name: str = 'empty_world'
#     robot_name: str = 'turtlebot4'
#     robot_radius: float = 0.3


# @dataclass(frozen=True)
# class PIDConfig:
#     path: str = 'line'
#     kp: np.ndarray = np.array([1, 1])
#     ki: np.ndarray = np.array([0, 0])
#     kd: np.ndarray = np.array([0, 0])
#     speed: float = 1



# def make_env(config: ExperimentConfig) -> gym.Env:
#     rclpy.init(args=None)
#     env_params = asdict(config.env)
#     env = gym.make(
#         env_params.pop('env_id'),
#         **env_params
#     )
#     # env = gym.wrappers.FlattenObservation(env)

def get_path(path_type: str):
    if path_type == "line":
        start_pos = (0, 0, 0) # (x, y, yaw)
        goal_pos = (10, 0, 0)
        path = np.vstack([
                10 * np.linspace(0, 100, 5000),
                0 * np.ones(5000)
            ])
        is_loop=False
    return start_pos, goal_pos, path, is_loop




    



def main():
    print("entered main")
    parser = argparse.ArgumentParser(
        description='Turtlebot4 Navigation with PID',
    )
    parser.add_argument("--pid-config")
    parser.add_argument("--world")

    args = parser.parse_args()
    print("args parsed")
    with open(args.pid_config, 'r') as file:
    # Use safe_load to convert the YAML content to a Python dictionary
        pid_config = yaml.safe_load(file)

    print("loaded_config")

    rclpy.init()
    env=gym.make('Turtlebot4Env-v0')

    start_pos, goal_pos, path = get_path(pid_config["path"])

    kp = pid_config["kp"]
    ki = pid_config["ki"]
    kd = pid_config["kd"]


    state = env.reset(options={"start_pos": start_pos, "goal_pos": goal_pos})
    print("env made")

        
    xy_err = [0]
    xy_prev = 0

    speed_err = [0]
    speed_prev=0
    done = False
    
    rewards = []
    observation, reward, terminated, truncated, info = env.step(np.ndarray([1, 0]))


    i = 0
    while (not terminated):
        # print(i)

        

        (h_x, h_y) = (x_at_H - np.array([x, y]))

        diff = np.arctan2(h_y, h_x)
        
        e = diff - psi
        e = wrap_angle(e)

        # if e == 0 or np.linalg.norm([v_x, v_y])<2:
        #     thrust = 1
        # else: 
        thrust = 1*(np.pi/10 - abs(e)) + (speed-v_x)

        prev = errors[-1]
        errors.append(e)
        
        prev2 = errors_2[-1]
        errors_2.append(thrust)
        

        steer = kp[0] * e + ki[0] * sum(errors) * 0.05 + kd[0] * (e- prev) / 0.05
        thrust = kp[1] * (thrust) + ki[1] * sum(errors_2) * 0.05 + kd[0] * (thrust-prev2) /0.05
 
        observation, reward, done, _ = rc_env.step([steer, thrust])
        rewards.append(reward)
        rc_env.render()

        
    print(sum(rewards))

    print("done")
    env.close()


if __name__ == '__main__':
    main()