import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
import numpy as np
import torch

def plot_rewards(csv):
	df = pd.read_csv(csv)
	# x-axis: iteration numbers
	iters = np.arange(1, len(df["rewards"]) + 1)
	plt.title(title)

	# plot raw rewards (faint)
	plt.plot(iters, df["rewards"], label="Avg Rewards")

	# add labels and legend
	plt.xlabel("Episode")
	plt.ylabel("Reward")
	plt.legend()
	plt.grid()
	plt.savefig(f'reinforcement_results/{title}.png')

def plot_reward_from_model(model_name, episodes_per_iter):
	cp_obj = torch.load(model_name, weights_only=False)
	total_ep_rewards = cp_obj["total_rewards"]
	print(f"Total episodes: {len(total_ep_rewards)}")
	iters = np.arange(1, len(total_ep_rewards) + 1)

	# plt.title(title)

	# # plot raw rewards (faint)
	# plt.plot(iters, total_ep_rewards, label="Total Rewards")

	# # add labels and legend
	# plt.xlabel("Episode")
	# plt.ylabel("Total Reward")
	# plt.legend()
	# plt.grid()
	# plt.savefig(f'{title}_totalreward.png')
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
	fig.suptitle(title)

	# --- Top plot: per episode + 5-episode rolling avg ---
	ax1.plot(iters, total_ep_rewards, alpha=0.3, label="Per episode")
	smoothed = pd.Series(total_ep_rewards).rolling(10).mean()
	ax1.plot(iters, smoothed, label="10-ep rolling avg", linewidth=2)

	num_iters = len(total_ep_rewards) // episodes_per_iter
	for i in range(1, num_iters):
		ax1.axvline(x=i * episodes_per_iter, color='r', alpha=0.2, linewidth=0.8)

	ax1.set_xlabel("Episode")
	ax1.set_ylabel("Total Reward")
	ax1.legend()
	ax1.grid()

	# --- Bottom plot: avg reward per iteration ---
	iter_avgs = []
	for i in range(num_iters):
		slice_ = total_ep_rewards[i * episodes_per_iter:(i + 1) * episodes_per_iter]
		iter_avgs.append(np.mean(slice_))

	ax2.plot(np.arange(1, num_iters + 1), iter_avgs, marker='o', label="Avg reward per iteration")
	ax2.set_xlabel("Iteration")
	ax2.set_ylabel("Avg Reward")
	ax2.legend()
	ax2.grid()

	plt.tight_layout()
	plt.savefig(f'{title}_totalreward_wavg2.png')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--k", type=int, default=5)
	parser.add_argument("--episodes", type=int)
	parser.add_argument("--max-steps", type=int)
	parser.add_argument("--input", type=str, default="csv")
	args = parser.parse_args()
	title = f"episodes:{args.episodes}-k:{args.k}-maxsteps:{args.max_steps}"
	title = "iter:9-episodes:50-k:5-maxsteps:201"
	#TODO: add check for if file name of that title exists?
	
	if args.input == "csv":
		plot_rewards(f"{title}.csv")	# plots rewards via csv
	elif args.input == "pt":
		plot_reward_from_model(f"{title}.pt", args.episodes)	# plots total rewards via saved model