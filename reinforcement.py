import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

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

GAMMA = 0.9
SCENARIOS = [
	"line",
	"loose_sin",
	"mid_sin",
	"tight_sin",
]
ORDER = [0, 1, 2, 3]
MIN_EP_FRAC = 0.2

class NN(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_size, 32),
			nn.ReLU(),
			nn.Linear(32, output_size),
			nn.Softmax(dim=-1)
		)
	
	def forward(self, x):
		return self.net(x)
	
	def action(self, state):
		state_t = torch.FloatTensor(state).unsqueeze(0)
		probs = self.forward(state_t)
		dist = Categorical(probs)
		action = dist.sample()
		log_prob = dist.log_prob(action)
		return action.item(), log_prob
	
def setup_scenario(ep_idx, total_eps): 
	# chooses scenario via returning idx for SCENARIO array
	# scenario_idx =  np.random.randint(len(SCENARIOS))
	# if ep_idx < (0.4 * total_eps):
	# 	scenario_idx = ORDER[ep_idx % len(ORDER)]

	# _, _, path = get_path(SCENARIOS[scenario_idx])
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
	# print(f"episode:{ep_idx} | scenario: {SCENARIOS[scenario_idx]} | start_pos: {start_pos} | goal_pos:{goal_pos}")
	return path, start_pos, goal_pos
	
def rewards_to_be(rewards):
	T = len(rewards)
	Q = [0.0] * T
	Q[-1] = rewards[-1]
	for t in reversed(range(T-1)):
		Q[t] = rewards[t] + GAMMA * Q[t+1]
	return Q

def plot_rewards(csv, title):
	df = pd.read_csv(csv)
	# x-axis: iteration numbers
	iters = np.arange(1, len(df["rewards"]) + 1)
	plt.title(title)

	# plot raw rewards (faint)
	plt.plot(iters, df["rewards"], label="Avg Rewards")

	# add labels and legend
	plt.xlabel("Iteration")
	plt.ylabel("Reward")
	plt.legend()
	plt.grid()
	plt.savefig(f'{title}.png')


def get_closest_index(path: np.ndarray, x: float, y: float):
	dist_arr = np.sqrt((path[0, :] - x)**2 + (path[1, :]-y)**2)
	return dist_arr.argmin(), dist_arr.min()

def calculate_curvature(path: np.ndarray, i: int, horizon: int):
	max_index = path.shape[1] - 1
	if i + horizon > max_index:
		horizon_index = max_index
	else:
		horizon_index = i + horizon
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
	# r_speed = -w[2] * (speed - target_speed)**2 # penalize speed deviation
	# r_progress = w[3]* speed * np.cos(heading_err) # reward forward progress
	r_dcte = -w[4] * dcte**2 # penalize growing lateral error

	# --- Sparse terminal rewards ---
	r_goal = +100.0 if done_goal else 0.0

	return r_cte + r_heading + r_dcte + r_goal
	
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
	parser.add_argument("--checkpoint", type=str, default=None)

	args = parser.parse_args()
	world = args.world
	episodes = args.episodes
	max_steps = args.max_steps
	k = args.k
	cp = args.checkpoint
		
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

	policy = NN(obs_dim, act_dim)
	optimizer = optim.Adam(policy.parameters(), lr=1e-2)

	avg_rewards_per_iteration = []
	total_ep_rewards = []

	avg_R = 0
	start_e = 0
	if cp is not None:
		cp_obj = torch.load(cp, weights_only=False)
		policy.load_state_dict(cp_obj["model_state_dict"])
		optimizer.load_state_dict(cp_obj["optimizer_state_dict"])
		start_e = cp_obj["episodes"]
		avg_rewards_per_iteration=cp_obj["avg_rewards"]
		chosen_pid = cp_obj["last_pid"]
		avg_R = avg_rewards_per_iteration[-1]
		total_ep_rewards = cp_obj["total_rewards"]

	pbar = tqdm(range(start_e, episodes), desc=f"{avg_R}")
	for e in pbar:
		pbar.set_description(f"{avg_R:.3f}")
		chosen_pid = -1

		path, start_pos, goal_pos = setup_scenario(e, episodes)
		env.reset(options={"start_pos": start_pos, "goal_pos": goal_pos})
		# print("env reset")

		ep_log_probs = []
		ep_Q = []
		ep_R = []

		yaw_err = [0]
		observation, reward, terminated, truncated, info = env.step(np.array([1, 0]))
		total_steps = 0
		dist = 0
		prev_dist = 0
		dx = 0
		dy = 0
		xy, yaw = get_robot_xy(env.env.env.env)

		while(not terminated and total_steps < max_steps):
			# return the cte, heading error, curvature, velocity, dcte/dt, lookahead err
			xy, yaw = get_robot_xy(env.env.env.env)
			prev_dist = dist
			closest_path_i, dist = get_closest_index(path, xy[0], xy[1])
			horizon_xy = get_horizon_xy(path, closest_path_i, 50)

			(dx, dy) = (horizon_xy - xy)
			diff = np.arctan2(dy, dx)
			curr_yaw_err = diff - yaw

			curr_speed = env.env.env.env.sensors._odom_msg.twist.twist.linear.x
			lookahead_err = -np.sin(yaw) * dx + np.cos(yaw) * dy
			dcte_dt = (dist - prev_dist)/0.05
			curvature = calculate_curvature(path, closest_path_i, 50)

			# print("choosing pid")
			chosen_pid, log_prob = policy.action(torch.FloatTensor([dist, curr_yaw_err, curvature, curr_speed, dcte_dt, lookahead_err, chosen_pid]))
			ep_log_probs.append(log_prob)
			# print("chose pid")


			kp = pids[chosen_pid]["kp"][0]
			ki = pids[chosen_pid]["ki"][0]
			kd = pids[chosen_pid]["kd"][0]
			speed = pids[chosen_pid]["speed"]
			horizon = pids[chosen_pid]["horizon"]

			segment_rewards = 0

			while (not terminated and total_steps < max_steps):
				# print("step: ", total_steps)
				xy, yaw = get_robot_xy(env.env.env.env)
				prev_dist = dist
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

				segment_rewards += compute_reward([dist, curr_yaw_err, curr_speed, speed, dist - prev_dist, terminated])
				if total_steps % k == 0: 
					break

			ep_R.append(segment_rewards)

		ep_Q = torch.FloatTensor(rewards_to_be(ep_R))  
		b = ep_Q.mean()
		st_dev = ep_Q.std()

		norm_Q = (ep_Q - b)/(st_dev + 1e-8)       
		
		# calculate loss
		loss = torch.tensor(0.0, requires_grad=True)
		for lp, q in zip(ep_log_probs, norm_Q):
			loss = loss + (-q * lp)
		
		loss = loss / len(ep_Q)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		avg_R = np.mean(ep_R)
		avg_rewards_per_iteration.append(avg_R)
		total_ep_rewards.append(sum(ep_R))

		# print(f"Episode: {e} | Avg Reward per episode per iteration (update): {avg_R} | loss: {loss.item()}")

	env.close()
	title = f"episodes:{episodes}-k:{k}-maxsteps:{max_steps}"

	torch.save({
		"episodes": episodes,
		"model_state_dict": policy.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"avg_rewards": avg_rewards_per_iteration,
		"total_rewards": total_ep_rewards,
		"last_pid": chosen_pid
		}, f"{title}.pt"
	)
	pd.DataFrame({"rewards": avg_rewards_per_iteration}).to_csv(f"{title}.csv")
	return policy, avg_rewards_per_iteration, title

if __name__ == "__main__":
	policy, rewards, title = main()
	

	# plot_rewards(f"{title}.csv", title)
