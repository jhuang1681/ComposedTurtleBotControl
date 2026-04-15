import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import copy

import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical

import argparse
from tqdm import tqdm

from get_path import get_path
from pid_controller import get_horizon_xy, get_robot_xy

from dataclasses import asdict, dataclass, field
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import rclpy

import tb4_drl_navigation.envs  # noqa: F401
import time
import yaml
from transforms3d.euler import quat2euler, euler2quat

from collections import deque
from reinforcement import get_path, get_closest_index, get_horizon_xy, get_robot_xy, get_state, compute_reward, calculate_curvature, rewards_to_be


class QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )
    
    def forward(self, x):
        return self.net(x)
    
    def action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        return self.forward(state_t).argmax()
    
def main():
    parser = argparse.ArgumentParser(
        description='Turtlebot4 Navigation with PID',
    )
    parser.add_argument("--pid-configs", nargs="+")
    parser.add_argument("--world")
    parser.add_argument("--k", type=int, default=5)
    # parser.add_argument("--episodes", type=int, default=100)
    # parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--cp")


    args = parser.parse_args()
    world = args.world
    # episodes = args.episodes
    # max_steps = args.max_steps
    k = args.k
    cp = args.cp


    pids = []  # list of pid configurations to train RL to choose from
    for pid_config in args.pid_configs:
        with open(pid_config, 'r') as file:
        # Use safe_load to convert the YAML content to a Python dictionary
            pids.append(yaml.safe_load(file))
    print("loaded_config")

    rclpy.init()
    print("rclpy init")

    env=gym.make('Turtlebot4Env-v0', world_name=world, map_path=Path(f"/workspaces/cs558_proj/maps/{world}.pgm"), yaml_path=Path(f"/workspaces/cs558_proj/maps/{world}.yaml"), shuffle_on_reset=False)
    print("env made")

    obs_dim = 7 # TODO change
    act_dim = len(args.pid_configs)

    Q = QNet(obs_dim, act_dim)

    optimizer = optim.Adam(Q.parameters(), lr=1e-2)

    avg_rewards_per_update = [0]
    total_ep_rewards = []
    chosen_pid = -1
    step_counter = 0
    if cp is not None:
        cp_obj = torch.load(cp, weights_only=False)
        Q.load_state_dict(cp_obj["model_state_dict"])
        optimizer.load_state_dict(cp_obj["optimizer_state_dict"])

    start_pos, goal_pos, path = get_path("", args.load_path)
    env.reset(options={"start_pos": start_pos, "goal_pos": goal_pos})

    integral_yaw_err = [0.0]
    prev_yaw_err = 0.0
    cte = 0.0
    prev_cte = 0.0

    observation, reward, terminated, truncated, info = env.step(np.array([1, 0]))
    total_steps_in_episode = 0
    curr_ep_reward = 0
    x_list = []
    y_list = []
    while(not terminated):
        # return the cte, heading error, curvature, velocity, dcte/dt, lookahead err
        state = get_state(env, prev_cte, path)
        xy, yaw = get_robot_xy(env)
        x_list.append(xy[0])
        y_list.append(xy[1])
        state.append(chosen_pid)

        prev_state = state

        chosen_pid = Q.action(torch.FloatTensor(state))

        kp = pids[chosen_pid]["kp"][0]
        ki = pids[chosen_pid]["ki"][0]
        kd = pids[chosen_pid]["kd"][0]
        speed = pids[chosen_pid]["speed"]
        horizon = pids[chosen_pid]["horizon"]

        segment_rewards = 0
        num_segments = 0
        while (not terminated):
            # print("step: ", total_steps)
            [cte, curr_yaw_err, curvature, curr_speed, dcte_dt, lookahead_err] = get_state(env, prev_cte, path)
            xy, yaw = get_robot_xy(env)
            x_list.append(xy[0])
            y_list.append(xy[1])

            integral_yaw_err[chosen_pid] += curr_yaw_err
            steer_yaw = kp * curr_yaw_err + ki * integral_yaw_err[chosen_pid] * 0.05 + kd * (curr_yaw_err - prev_yaw_err) / 0.05
            steer = np.clip(steer_yaw, -np.pi/2, np.pi/2)

            _, _, terminated, _, _ = env.step(np.array([speed, steer]))
            total_steps_in_episode += 1
            step_counter += 1

            segment_rewards += compute_reward([cte, curr_yaw_err, curr_speed, speed, cte - prev_cte, terminated])
            prev_yaw_err = curr_yaw_err
            prev_cte = cte 
            if total_steps_in_episode % k == 0: 
                break

        num_segments += 1
        curr_ep_reward += segment_rewards

        xy, yaw = get_robot_xy(env)
        if xy[0] - 5 > goal_pos[0]:
            terminated=True
            print("Failed")

    df = pd.DataFrame({"x": x_list, "y": y_list})
    df.to_csv(f"{pid_config["path"]}_{args.load_path}.csv")
    print("done")
    env.close()