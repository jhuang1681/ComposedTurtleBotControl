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


GAMMA = 0.9
MIN_EP_FRAC = 0.2

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.replay_buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.replay_buffer)

    def append(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # samples = np.array(random.sample(self.replay_buffer, batch_size))
        samples = random.sample(self.replay_buffer, batch_size)
        # states, actions, rewards, next_states, dones = samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3], samples[:, 4]
        states, actions, rewards, next_states, dones = zip(*samples)
        # return states, actions, rewards, next_states, dones
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

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

    
def setup_scenario(ep_idx, total_eps): 
    _, _, path = get_path("random")

    max_start_idx = int((1.0 - MIN_EP_FRAC) * 5000) # track_len in get_path.py hardcoded as 5000
    start_idx = np.random.randint(0, max_start_idx)
    min_goal_idx = start_idx + int(MIN_EP_FRAC * 5000)
    goal_idx = np.random.randint(min_goal_idx, 4999)

    dx = path[0, start_idx + 1] - path[0, start_idx]
    dy = path[1, start_idx + 1] - path[1, start_idx]
    start_pos = np.array([path[0, start_idx], path[1, start_idx], np.arctan2(dy, dx)])
    dx = path[0, goal_idx + 1] - path[0, goal_idx]
    dy = path[1, goal_idx + 1] - path[1, goal_idx]
    goal_pos = np.array([path[0, goal_idx], path[1, goal_idx], np.arctan2(dy, dx)])
    return path, start_pos, goal_pos
    
def rewards_to_be(rewards):
    T = len(rewards)
    Q = [0.0] * T
    Q[-1] = rewards[-1]
    for t in reversed(range(T-1)):
        Q[t] = rewards[t] + GAMMA * Q[t+1]
    return Q

def get_closest_index(path: np.ndarray, x: float, y: float):
    dist_arr = np.sqrt((path[0, :] - x)**2 + (path[1, :]-y)**2)
    return dist_arr.argmin(), dist_arr.min()

def calculate_curvature(path: np.ndarray, i: int, horizon: int):
    max_index = path.shape[1] - 1
    i = np.clip(i, 1, max_index - 1)
    horizon_index = i + horizon
    horizon_index = np.clip(horizon_index, 1, max_index - 1)

    path_heading_i = np.arctan2(path[1, i+1] - path[1, i-1], path[0, i+1] - path[0, i-1])
    path_heading_horizon = np.arctan2(path[1, horizon_index+1] - path[1, horizon_index-1], path[0, horizon_index+1] - path[0, horizon_index-1])

    d_heading = path_heading_i - path_heading_horizon
    if d_heading > np.pi:
        d_heading = d_heading - 2*np.pi
    elif d_heading < -np.pi:
        d_heading = d_heading + 2*np.pi

    dist = np.sqrt((path[0, horizon_index] - path[0, i])**2 + (path[1, horizon_index]-path[1, i])**2)

    return d_heading/dist

def compute_reward(state):
    cte, heading_err, speed, target_speed, dcte, done_goal = state
    w = [2, 1, 0.5, 1, 0.5]
    r_cte = -w[0] * cte**2 # penalize lateral deviation
    r_heading = -w[1] * heading_err**2 # penalize misalignment
    r_dcte = -w[4] * dcte**2 # penalize growing lateral error

    # --- Sparse terminal rewards ---
    r_goal = +100.0 if done_goal else 0.0

    return r_cte + r_heading + r_dcte + r_goal
    
def get_state(env, prev_cte, path):
    xy, yaw = get_robot_xy(env.env.env.env)
    closest_path_i, cte = get_closest_index(path, xy[0], xy[1])
    horizon_xy = get_horizon_xy(path, closest_path_i, 50)
    (dx, dy) = (horizon_xy - xy)
    diff = np.arctan2(dy, dx)
    curr_yaw_err = diff - yaw
    curr_speed = env.env.env.env.sensors._odom_msg.twist.twist.linear.x
    lookahead_err = -np.sin(yaw) * dx + np.cos(yaw) * dy
    dcte_dt = (cte - prev_cte)/0.05
    curvature = calculate_curvature(path, closest_path_i, 50)

    if curr_yaw_err > np.pi:
        curr_yaw_err = curr_yaw_err - 2*np.pi
    elif curr_yaw_err < -np.pi:
        curr_yaw_err = curr_yaw_err + 2*np.pi

    return [cte, curr_yaw_err, curvature, curr_speed, dcte_dt, lookahead_err]

def main():
    print("entered main")
    parser = argparse.ArgumentParser(
        description='Turtlebot4 Navigation with PID',
    )
    parser.add_argument("--pid-configs", nargs="+")
    parser.add_argument("--world")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--update_target", type=int, default=500)
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()
    world = args.world
    episodes = args.episodes
    max_steps = args.max_steps
    k = args.k
    cp = args.checkpoint
    update_target = args.update_target

    epsilon = 1
    epsilon_min = 0.05
    epsilon_decay = 0.99
        
    print("args parsed")

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
    Q_fixed = copy.deepcopy(Q)

    replay_buffer = ReplayBuffer(1000)

    optimizer = optim.Adam(Q.parameters(), lr=1e-2)

    avg_rewards_per_update = [0]
    total_ep_rewards = []
    chosen_pid = -1
    start_e = 0
    step_counter = 0
    if cp is not None:
        cp_obj = torch.load(cp, weights_only=False)
        Q.load_state_dict(cp_obj["model_state_dict"])
        optimizer.load_state_dict(cp_obj["optimizer_state_dict"])
        start_e = cp_obj["episodes"]
        avg_rewards_per_update=cp_obj["avg_rewards"]
        chosen_pid = cp_obj["last_pid"]
        avg_R = avg_rewards_per_update[-1]
        total_ep_rewards = cp_obj["total_rewards"]

    Q_update_count = 0

    paths = []
    paths_traveled = []
    pbar = tqdm(range(start_e, episodes), desc=f"{avg_rewards_per_update[-1]}")
    for e in pbar:
        pbar.set_description(f"{avg_rewards_per_update[-1]:.3f}")
        path, start_pos, goal_pos = setup_scenario(e, episodes)
        paths.append(path)
        env.reset(options={"start_pos": start_pos, "goal_pos": goal_pos})
        # print("env reset")

        path_traveled = [get_robot_xy(env.env.env.env)[0]]
        integral_yaw_err = [0.0 for _ in range(len(pids))]
        prev_yaw_err = 0.0
        cte = 0.0
        prev_cte = 0.0

        observation, reward, terminated, truncated, info = env.step(np.array([1, 0]))
        total_steps_in_episode = 0
        

        curr_ep_reward = 0
        
        while(not terminated and total_steps_in_episode < max_steps):
            # return the cte, heading error, curvature, velocity, dcte/dt, lookahead err
            state = get_state(env, prev_cte, path)
            state.append(chosen_pid)
            path_traveled.append(get_robot_xy(env.env.env.env)[0])
            prev_state = state

            # print("choosing pid")
            # if np.random.random() < (1-epsilon):
            if np.random.random() < epsilon: # CHANGED
                chosen_pid = np.random.randint(low=0, high=len(pids))
            else:
                chosen_pid = Q.action(torch.FloatTensor(state))
            # print("chose pid")

            kp = pids[chosen_pid]["kp"][0]
            ki = pids[chosen_pid]["ki"][0]
            kd = pids[chosen_pid]["kd"][0]
            speed = pids[chosen_pid]["speed"]
            horizon = pids[chosen_pid]["horizon"]

            segment_rewards = 0
            num_segments = 0
            while (not terminated and total_steps_in_episode < max_steps):
                # print("step: ", total_steps)
                [cte, curr_yaw_err, curvature, curr_speed, dcte_dt, lookahead_err] = get_state(env, prev_cte, path)
                path_traveled.append(get_robot_xy(env.env.env.env)[0])


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

            next_state = get_state(env, cte, path)
            next_state.append(chosen_pid)

            replay_buffer.append(prev_state, chosen_pid, segment_rewards, next_state, terminated)
            
            batch_size = 64
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states_t      = torch.FloatTensor(states)
                actions_t     = torch.LongTensor(actions)
                rewards_t     = torch.FloatTensor(rewards)
                next_states_t = torch.FloatTensor(next_states)
                dones_t       = torch.FloatTensor(dones)
                with torch.no_grad():
                    fixed_q_vals = Q_fixed(next_states_t)
                    max_fixed_q = fixed_q_vals.max(dim=1).values

                targets = rewards_t + GAMMA * max_fixed_q * (1-dones_t)

                q_vals = Q(states_t)
                q_chosen_pids = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze()

                loss = ((q_chosen_pids - targets) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            last_pid = chosen_pid

            if step_counter % update_target > Q_update_count:
                Q_fixed = copy.deepcopy(Q)
                Q_update_count += 1

        total_ep_rewards.append(curr_ep_reward)
        avg_rewards_per_update.append(curr_ep_reward / num_segments)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        paths_traveled.append(path_traveled)

        # title = f"episodes:{episodes}-k:{k}-maxsteps:{max_steps}"
        # if (e + 1) % 100 == 0:
        #     torch.save({
        #         "episodes": e + 1,
        #         "model_state_dict": Q.state_dict(),
        #         "fixed_model_state_dict": Q_fixed.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "avg_rewards": avg_rewards_per_update,
        #         "total_rewards": total_ep_rewards,
        #         "last_pid": last_pid,
        #         "step_counter": step_counter,
        #         "epsilon": epsilon
        #     }, f"{title}_ep{e+1}.pt")
        #     print(f"Checkpoint saved at episode {e+1}", flush=True)

        curr_speed = env.env.env.env.sensors._odom_msg.twist.twist.linear.x
        # print(f"Episode: {e} | Avg Reward per episode per iteration (update): {avg_R} | loss: {loss.item()}")

    env.close()
    title = f"visualizer_episodes:{episodes}-k:{k}-maxsteps:{max_steps}"

    torch.save({
        "episodes": episodes,
        "paths": paths,
        "paths_traveled": paths_traveled,
        # "model_state_dict": Q.state_dict(),
        # "fixed_model_state_dict": Q_fixed.state_dict(),
        # "optimizer_state_dict": optimizer.state_dict(),
        # "avg_rewards": avg_rewards_per_update,
        # "total_rewards": total_ep_rewards,
        # "last_pid": last_pid,
        # "step_counter": step_counter,
        # "epsilon": epsilon
        }, f"{title}.pt"
    )
    
    pd.DataFrame({"rewards": avg_rewards_per_update}).to_csv(f"{title}.csv")
    return Q, avg_rewards_per_update, title

if __name__ == "__main__":
    Q, rewards, title = main()