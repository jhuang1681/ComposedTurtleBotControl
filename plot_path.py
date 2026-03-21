import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from get_path import get_path

print("imports done")


parser = argparse.ArgumentParser()
parser.add_argument("--path-type")
parser.add_argument("--slope", type=float)
parser.add_argument("--data-file")
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
plt.plot(data["x"], data["y"], 'o-', label="CSV Data")

print("plt plotted")


plt.legend()
plt.grid()
# plt.show()
print("b4 save fig")
plt.savefig("course.png")