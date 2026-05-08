import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
from get_path import get_path
import torch
from pid_controller import get_closest_index, get_horizon_xy

def plot_path():
    # Plot both
    total = data["rewards"].sum()
    plt.plot(path[0,:], path[1,:], 'o-', markersize=1, label="Desired Curve")
    plt.plot(data["x"], data["y"], 'o-', markersize=2, label="CSV Data")
    plt.text(0.1, 0.9, f"Total Reward = {total:.3f}", transform=plt.gca().transAxes)
    plt.legend()
    plt.grid()
    plt.savefig(f"{file_name}_path.png")
    print("saved path plot")

def plot_speed():
    fig, ax = plt.subplots()

    ax.plot(data.index, data["speed"], 'o-', label="speed")

    ax.legend()
    ax.grid()
    fig.savefig(f"{file_name}_speed.png")
    print("saved speed plot")

def plot_xy_err():
    fig, ax = plt.subplots()
    mse = (data["xy_err"] ** 2).mean()
    print(mse)
    ax.text(0.1, 0.9, f"MSE = {mse:.7f}", transform=ax.transAxes)
    ax.plot(data.index, data["xy_err"], 'o-', markersize=2, label="xy distance err")
    ax.legend()
    ax.grid()
    fig.savefig(f"{file_name}_xy_err.png")
    print("saved xy err plot")

def plot_cte_err():
    fig, ax = plt.subplots()
    actual_ctes = []
    closest_path_i = 0
    for i, row in data.iterrows():
        closest_path_i, dist = get_closest_index(path, row.x, row.y, closest_path_i, True)
        dx_0 = path[0, closest_path_i] - row.x
        dy_0 = path[1, closest_path_i] - row.y
        # print(path[0, closest_path_i], path[1, closest_path_i], row.x, row.y)
        actual_cte = -np.sin(row.yaw) * dx_0 + np.cos(row.yaw) * dy_0
        actual_ctes.append(actual_cte)

    actual_ctes = np.array(actual_ctes)
    mse = (actual_ctes ** 2).mean()
    ax.text(0.1, 0.9, f"MSE = {mse:.7f}", transform=ax.transAxes)
    ax.plot(data.index, actual_ctes, 'o-', markersize=2, label="cte err")

    ax.set_ylim(-0.8, 0.8)
    ax.legend()
    ax.grid()
    fig.savefig(f"{file_name}_CTE.png")
    print("saved cte err plot")

def plot_cte():
    fig, ax = plt.subplots()
    mse = (data["cte"] ** 2).mean()
    ax.text(0.1, 0.9, f"MSE = {mse:.7f}", transform=ax.transAxes)
    ax.plot(data.index, data["cte"], 'o-', markersize=2, label="cte")

    ax.legend()
    ax.grid()
    fig.savefig(f"{file_name}_actual_cte.png")
    print("saved cte plot")

def plot_reward_from_model(model_name):
    cp_obj = torch.load(model_name, weights_only=False)
    ep = cp_obj["episodes"]
    paths = cp_obj["paths"]
    paths_t = cp_obj["paths_traveled"]

    for i, path in enumerate(paths):
        plt.clf()
        plt.plot(path[0,:], path[1,:], label="Desired path")

        data = np.array(paths_t[i])
        plt.plot(data[:, 0], data[:, 1], 'o-', markersize=2, label="CSV Data")

        plt.legend()
        plt.grid()
        plt.savefig(f"{model_name}_{i}_path.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-type", type=str, default="")
    parser.add_argument("--slope", type=float, default=0.0)
    parser.add_argument("--data-file")
    parser.add_argument("--yaml", type=str, default="configs/line.yml")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--load-path", type = str, default = None)
    args = parser.parse_args()

    if args.model_name is None:
        start, goal, path, _ = get_path(args.path_type, args.load_path)
        data = pd.read_csv(args.data_file)
        with open(args.yaml, 'r') as file:
                pid_config = yaml.safe_load(file)

        kp = pid_config["kp"]
        ki = pid_config["ki"]
        kd = pid_config["kd"]

        file_name = f"{args.path_type}_{kp}_{ki}_{kd}_{pid_config["horizon"]}_{pid_config["speed"]}_{args.load_path}"
        if args.load_path is not None:
            file_name=f"{args.load_path}_{args.data_file}"
            print(file_name)
            plot_path()
            # plot_cte()
            # plot_speed()
            # plot_xy_err()
            plot_cte_err()
        else:
            plot_path()
            # plot_xy_err()
            # plot_cte()
            plot_cte_err()
    else:
        plot_reward_from_model(args.model_name)
