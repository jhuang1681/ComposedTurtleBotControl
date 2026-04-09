import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
from get_path import get_path

print("imports done")


parser = argparse.ArgumentParser()
parser.add_argument("--path-type")
parser.add_argument("--slope", type=float)
parser.add_argument("--data-file")
parser.add_argument("--yaml")
args = parser.parse_args()

# Sine curve
print("getting path")
start, goal, path, is_loop = get_path(args.path_type, args.slope)
print("path got")
# CSV data
data = pd.read_csv(args.data_file)
print("data got")

print(path.shape)
print(len(data))

# Plot both
print("plt plotting")
plt.plot(path[0,:], path[1,:], label="Sine Curve")
plt.plot(data["x"], data["y"], 'o-', markersize=2, label="CSV Data")

print("plt plotted")

with open(args.yaml, 'r') as file:
        pid_config = yaml.safe_load(file)

print("loaded yaml")

kp = pid_config["kp"]
ki = pid_config["ki"]
kd = pid_config["kd"]

file_name = f"headless_outputs/{args.path_type}_{kp}_{ki}_{kd}_{pid_config["slope"]}_{pid_config["speed"]}"

plt.legend()
plt.grid()
# plt.show()
print("b4 save fig")
plt.savefig(f"{file_name}_path.png")

fig, ax = plt.subplots()

# # Plot both
print("plt plotting")
# plt.plot(path[0,:], path[1,:], label="Sine Curve")
ax.plot(data.index, data["speed"], 'o-', label="speed")

print("plt plotted")


ax.legend()
ax.grid()
# plt.show()
print("b4 save fig")
fig.savefig(f"{file_name}_speed.png")

fig, ax = plt.subplots()

# Plot both
print("plt plotting")
# plt.plot(path[0,:], path[1,:], label="Sine Curve")
ax.plot(data.index, data["xy_err"], 'o-', markersize=2, label="err Data")

print("plt plotted")


ax.legend()
ax.grid()
# plt.show()
print("b4 save fig")
fig.savefig(f"{file_name}_xy_err.png")


# fig, ax = plt.subplots()


#  #Plot both
# print("plt plotting")
# ax.plot(data.index, data["speed_err"], 'o-', label="speed_err Data")

# print("plt plotted")


# ax.legend()
# ax.grid()
# # plt.show()
# print("b4 save fig")
# fig.savefig(f"{args.path_type}_speed_err.png")