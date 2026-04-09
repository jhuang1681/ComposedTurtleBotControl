import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
import numpy as np
title = "k:5-episodes:5-maxsteps:10"
filename = f"{title}.csv"

def plot_rewards(csv):
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

if __name__ == "__main__":
	plot_rewards(filename)