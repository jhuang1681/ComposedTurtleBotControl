import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import time

import argparse

from get_path import get_path
from pid_controller import get_robot_xy, get_horizon_xy

from pathlib import Path

import pandas as pd
import rclpy

import tb4_drl_navigation.envs  # noqa: F401
import yaml

from reinforcement import get_closest_index, get_state, compute_reward, calculate_curvature


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

    chosen_pid = -1
    step_counter = 0
    if cp is not None:
        cp_obj = torch.load(cp, weights_only=False)
        Q.load_state_dict(cp_obj["model_state_dict"])
        optimizer.load_state_dict(cp_obj["optimizer_state_dict"])

    start_pos, goal_pos, path = get_path("", args.load_path)
    env.reset(options={"start_pos": start_pos, "goal_pos": goal_pos})

    integral_yaw_err = 0.0
    prev_yaw_err = 0.0
    cte = 0.0
    prev_cte = 0.0
    actual_ctes = []

    start = time.time()

    observation, reward, terminated, truncated, info = env.step(np.array([1, 0]))
    total_steps_in_episode = 0
    curr_ep_reward = 0
    x_list = []
    y_list = []
    cte_list = []
    segreward_list = []
    while(not terminated):
        # return the cte, heading error, curvature, velocity, dcte/dt, lookahead err
        state = get_state(env, prev_cte, path)
        xy, yaw = get_robot_xy(env.env.env.env)
        x_list.append(xy[0])
        y_list.append(xy[1])
        cte_list.append(state[0])

        closest_path_i, dist = get_closest_index(path, xy[0], xy[1])
        (dx_0, dy_0) = get_horizon_xy(path, closest_path_i,0)
        actual_cte = -np.sin(yaw) * dx_0 + np.cos(yaw) * dy_0
        actual_ctes.append(actual_cte)
        segreward_list.append(0.0) 
        state.append(chosen_pid)

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
            cte_list.append(cte)
            xy, yaw = get_robot_xy(env.env.env.env)
            x_list.append(xy[0])
            y_list.append(xy[1])

            closest_path_i, dist = get_closest_index(path, xy[0], xy[1])
            (dx_0, dy_0) = get_horizon_xy(path, closest_path_i,0)
            actual_cte = -np.sin(yaw) * dx_0 + np.cos(yaw) * dy_0
            actual_ctes.append(actual_cte)

            integral_yaw_err += curr_yaw_err
            steer_yaw = kp * curr_yaw_err + ki * integral_yaw_err * 0.05 + kd * (curr_yaw_err - prev_yaw_err) / 0.05
            steer = np.clip(steer_yaw, -np.pi/2, np.pi/2)

            _, _, terminated, _, _ = env.step(np.array([speed, steer]))
            
            total_steps_in_episode += 1
            step_counter += 1

            temp_reward = compute_reward([cte, curr_yaw_err, curr_speed, cte - prev_cte, terminated])
            segment_rewards += temp_reward
            segreward_list.append(temp_reward)
            prev_yaw_err = curr_yaw_err
            prev_cte = cte 
            if total_steps_in_episode % k == 0: 
                break

        num_segments += 1

        xy, yaw = get_robot_xy(env.env.env.env)

        if xy[0] - 5 > goal_pos[0]:
            terminated=True
            print("Failed")


    df = pd.DataFrame({"x": x_list, "y": y_list, "cte_err": cte_list, "cte": actual_ctes, "rewards": segreward_list})
    df.to_csv("test_rl_path_ddqn2-1.csv")
    print("done")
    env.close()
    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()