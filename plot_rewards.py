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

def plot_reward_from_model(model_name):
	cp_obj = torch.load(model_name, weights_only=False)
	start_e = cp_obj["episodes"]
	# avg_rewards_per_iteration=cp_obj["avg_rewards"]
	# chosen_pid = cp_obj["last_pid"]
	# avg_R = avg_rewards_per_iteration[-1]
	total_ep_rewards = cp_obj["total_rewards"]
	print(len(total_ep_rewards))
	iters = np.arange(1, len(total_ep_rewards) + 1)

	plt.title(title)

	# plot raw rewards (faint)
	plt.plot(iters, total_ep_rewards, label="Total Rewards")

	# add labels and legend
	plt.xlabel("Episode")
	plt.ylabel("Total Reward")
	plt.legend()
	plt.grid()
	plt.savefig(f'reinforcement_results/{title}_totalreward.png')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--k", type=int)
	parser.add_argument("--episodes", type=int)
	parser.add_argument("--max-steps", type=int)
	parser.add_argument("--input", type=str, default="csv")
	args = parser.parse_args()
	title = f"episodes:{args.episodes}-k:{args.k}-maxsteps:{args.max_steps}"
	#TODO: add check for if file name of that title exists?
	
	if args.input == "csv":
		plot_rewards(f"{title}.csv")	# plots rewards via csv
	elif args.input == "pt":
		plot_reward_from_model(f"{title}.pt")	# plots total rewards via saved model