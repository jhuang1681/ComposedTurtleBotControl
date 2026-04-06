import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import argparse

GAMMA = 0.9
N_ITERATIONS = 200
N_EPISODES = 40
# SEED = 42

class NN(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_size, 32),
			nn.Tanh(),
			nn.Linear(32, output_size),
		)
		self.log_std = nn.Parameter(torch.ones(output_size) * -0.5) #multiply with constant?
	
	def forward(self, x):
		miu = self.net(x)
		sd = torch.exp(self.log_std.clamp(min=-3.45)) 
		cov = torch.diag(sd ** 2)
		dist = torch.distributions.MultivariateNormal(miu, cov)
		return dist
	
	def action(self, x):
		dist = self.forward(x)
		action = dist.sample()
		log_prob = dist.log_prob(action)
		return action, log_prob
	
def episode_rollout(env, policy):
	state = env.reset()
	log_probs = []
	rewards = []
	done = False

	while not done:
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

def train():
	env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=True)
	# env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
	# torch.manual_seed(SEED)
	# env.seed(SEED)

	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]
	test_state = env.reset()
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

if __name__ == "__main__":
	# policy = NN(8, 2)
	# policy.load_state_dict(torch.load("policy.pt"))
	policy, rewards = train()
	torch.save(policy.state_dict(), "policy.pt")
	plot_rewards(rewards, title=f"Iterations:{N_ITERATIONS}-Episodes:{N_EPISODES}")
