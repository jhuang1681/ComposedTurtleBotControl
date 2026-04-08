import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical

import argparse

from get_path import get_path
from pid_controller import get_horizon_xy, get_robot_xy

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import rclpy

import tb4_drl_navigation.envs  # noqa: F401
import torch
import torch.nn as nn
import time
import yaml
from transforms3d.euler import quat2euler, euler2quat

GAMMA = 0.9
N_ITERATIONS = 200
N_EPISODES = 40
# SEED = 42

class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLu(),
            nn.Linear(32, output_size),
        )
    
    def forward(self, x):
        return self.net(x)
    
    def action(self, state):
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action, log_prob
    
def get_training_path() -> np.ndarray: # return vstack of x, y pos
    pass

def get_start_goal_pos() -> tuple[np.ndarray, np.ndarray]: # return tuple of (x, y, yaw) start and end
    pass
    
def run_k_steps(env, pid_config, k):
    state = env.reset()
    kp = pid_config["kp"]
    ki = pid_config["ki"]
    kd = pid_config["kd"]
    speed = pid_config["speed"]
    horizon = pid_config["horizon"]

    log_probs = []
    rewards = []
    done = False

    for i in range(k):
        state_t = torch.FloatTensor(state)
        action, log_prob = policy.action(state_t)
        state, reward, done, _ = env.step(action.detach().numpy())
        log_probs.append(log_prob)
        rewards.append(reward)
    
    return log_probs, np.array(rewards, dtype=np.float32)

def rewards_to_be(rewards):
    T = len(rewards)
    Q = [0.0] * T
    Q[-1] = rewards[-1]
    for t in reversed(range(T-1)):
        Q[t] = rewards[t] + GAMMA * Q[t+1]
    return Q

def train(env):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # print(f"obs_dim from space: {obs_dim}, actual state shape: {np.array(test_state).shape}")
    obs_dim = np.array(test_state).shape[0]  # use the real shape, not the declared one

    policy = NN(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    avg_rewards_per_iteration = []

    for iter in range(0, N_ITERATIONS):
        iter_log_probs = []
        iter_Q = []
        iter_R = []

        for _ in range(N_EPISODES):
            log_probs, rewards = episode_rollout(env, policy)

            iter_log_probs.append(log_probs)
            iter_Q.append(rewards_to_be(rewards))
            iter_R.append(rewards.sum())

        all_Q = torch.FloatTensor([q for episode_Q in iter_Q for q in episode_Q])
        b = all_Q.mean()
        stdv = all_Q.std() + 1e-8

        idx = 0
        norm_Q_per_episode = []
        for episode_Q in iter_Q:
            T = len(episode_Q)
            norm_Q_per_episode.append((all_Q[idx: idx+T]-b)/stdv)
            idx += T

        # calculate loss
        loss = torch.tensor(0.0, requires_grad=True)

        for log_probs, Q_ep in zip(iter_log_probs, norm_Q_per_episode):
            for log_prob, q in zip(log_probs, Q_ep):
                loss = loss + (-q * log_prob)

        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_R = np.mean(iter_R)
        avg_rewards_per_iteration.append(avg_R)

        print(f"Iteration: {iter} | Avg Reward: {avg_R} | loss: {loss.item()}")

    env.close()
    return policy, avg_rewards_per_iteration

def plot_rewards(avg_rewards, title):
    # x-axis: iteration numbers
    iters = np.arange(1, len(avg_rewards) + 1)
    plt.title(title)

    # plot raw rewards (faint)
    plt.plot(iters, avg_rewards, label="Avg Rewards")

    # add labels and legend
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    plt.savefig(f'{title}.png')
    plt.show()

def get_features():
    pass

def get_closest_index(path: np.ndarray, x: float, y: float):
    dist_arr = np.sqrt((path[0, :] - x)**2 + (path[1, :]-y)**2)
    return dist_arr.argmin(), dist_arr.min()

def main():
    print("entered main")
    parser = argparse.ArgumentParser(
        description='Turtlebot4 Navigation with PID',
    )
    parser.add_argument("--pid-configs", nargs="+")
    parser.add_argument("--world")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=50)

    args = parser.parse_args()
    world = args.world
    episodes = args.episodes
    max_steps = args.max_steps
    k = args.k
    print("args parsed")

    pids = []  # list of pid configurations to train RL to choose from
    for pid_config in args.pid_configs:
        with open(pid_config, 'r') as file:
        # Use safe_load to convert the YAML content to a Python dictionary
                pids.append(yaml.safe_load(file))

    print("loaded_config")

    rclpy.init()
    print("rclpy init")

    env=gym.make('Turtlebot4Env-v0', world_name=world, shuffle_on_reset=False)
    print("env made")

    path = get_training_path()

    obs_dim = 7 # TODO change
    act_dim = len(args.pid_configs)

    policy = NN(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    avg_rewards_per_iteration = []

    chosen_pid = -1
    for e in range(0, episodes):
        ep_log_probs = []
        ep_Q = []
        ep_R = []

        yaw_err = []
        observation, reward, terminated, truncated, info = env.step(np.array([1, 0]))
        total_steps = 0
        while(not terminated and total_steps < max_steps):
            # return the cte, heading error, curvature, velocity, dcte/dt, lookahead err
            cte, heading_err, curvature, velocity, dcte_dt, lookahead_err = get_features()

            chosen_pid, log_prob = policy.action(torch.FloatTensor([cte, heading_err, curvature, velocity, dcte_dt, lookahead_err, chosen_pid]))
            ep_log_probs.append(log_prob)


            kp = pids[chosen_pid]["kp"]
            ki = pids[chosen_pid]["ki"]
            kd = pids[chosen_pid]["kd"]
            speed = pids[chosen_pid]["speed"]
            horizon = pids[chosen_pid]["horizon"]

            segment_rewards = 0

            while (total_steps % k != 0 and not terminated):
                xy, yaw = get_robot_xy(env.env.env.env)
                closest_path_i, dist = get_closest_index(path, xy[0], xy[1])
                horizon_xy = get_horizon_xy(path, closest_path_i, horizon)

                (dx, dy) = (horizon_xy - xy)
                diff = np.arctan2(dy, dx)
                curr_yaw_err = diff - yaw

                if curr_yaw_err > np.pi:
                    curr_yaw_err = curr_yaw_err - 2*np.pi
                elif curr_yaw_err < -np.pi:
                    curr_yaw_err = curr_yaw_err + 2*np.pi


                prev_yaw_err = yaw_err[-1]
                yaw_err.append(curr_yaw_err)

                steer_yaw = kp * curr_yaw_err + ki * sum(yaw_err) * 0.05 + kd * (curr_yaw_err - prev_yaw_err) / 0.05
                steer = np.clip(steer_yaw, -np.pi/2, np.pi/2)

                _, _, terminated, _, _ = env.step(np.array([speed, steer]))
                total_steps += 1
                segment_rewards += calculate_reward()

            ep_R.append(segment_rewards)

        ep_Q = torch.FloatTensor(rewards_to_be(ep_R))  
        b = ep_Q.mean()
        st_dev = ep_Q.std()

        norm_Q = (ep_Q - b)/(st_dev + 1e-8)       
        
        # calculate loss
        loss = torch.tensor(0.0, requires_grad=True)
        for lp, q in zip(ep_log_probs, norm_Q):
            loss = loss + (-q * log_prob)
        
        loss = loss / len(ep_Q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_R = np.mean(ep_R)
        avg_rewards_per_iteration.append(avg_R)

        print(f"Episode: {e} | Avg Reward per episode per iteration (update): {avg_R} | loss: {loss.item()}")

    env.close()
    return policy, avg_rewards_per_iteration

        

    



if __name__ == "__main__":
    main()