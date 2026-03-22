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
# from stable_baselines3 import SAC
# from stable_baselines3.common.callbacks import (
#     BaseCallback,
#     CheckpointCallback,
#     EvalCallback
# )
# from stable_baselines3.common.monitor import Monitor
import tb4_drl_navigation.envs  # noqa: F401
import torch
import torch.nn as nn
import time
import yaml
from transforms3d.euler import quat2euler, euler2quat
from get_path import get_path


# @dataclass(frozen=True)
# class EnvConfig:
#     env_id: str = 'Turtlebot4Env-v0'
#     world_name: str = 'empty_world'
#     robot_name: str = 'turtlebot4'
#     robot_radius: float = 0.3


# @dataclass(frozen=True)
# class PIDConfig:
#     path: str = 'line'
#     kp: np.ndarray = np.array([1, 1])
#     ki: np.ndarray = np.array([0, 0])
#     kd: np.ndarray = np.array([0, 0])
#     speed: float = 1



# def make_env(config: ExperimentConfig) -> gym.Env:
#     rclpy.init(args=None)
#     env_params = asdict(config.env)
#     env = gym.make(
#         env_params.pop('env_id'),
#         **env_params
#     )
#     # env = gym.wrappers.FlattenObservation(env)

# def get_path(path_type: str):
#     if path_type == "line":
#         start_pos = np.array([0, 0, 0], dtype=np.float64) # (x, y, yaw)
#         goal_pos = np.array([10, 0, 0], dtype=np.float64)
#         path = np.vstack([
#                 10 * np.linspace(0, 100, 5000),
#                 0 * np.ones(5000)
#             ])
#         is_loop=False
#     return start_pos, goal_pos, path, is_loop


# TODO cite racecar
def get_closest_index(path: np.ndarray, x: float, y: float, start_ind: float, is_loop: bool):
    best_dist = np.inf
    
    ind_s = [i for i in range(start_ind)]
    ind_e = [start_ind + i for i in range(path.shape[1])]

    if not is_loop:
        indices = ind_e
    else:
        indices = [i for a in [ind_e, ind_s] for i in a]

    for i in indices:
        path_x = path[0, i]
        path_y = path[1, i]

        dist = np.sqrt((path_x - x)**2 + (path_y - y)**2)

        if dist < best_dist:
            best_dist = dist
        else:
            break
    return i, dist

def get_horizon_xy(path: np.ndarray, current_index: float, horizon: int):
    max_index = path.shape[1] - 1
    
    if current_index + horizon > max_index:
        index =  max_index
    else:
        index = current_index + horizon

    return np.array([path[0, index], path[1, index]])

def get_robot_xy(env):
    # Developed based on turtlebot4 _get_odom code
    # Get current pose
    pose_stamped = env.sensors.get_latest_pose_stamped()
    agent_pose = pose_stamped.pose

    # Extract positions
    agent_x = agent_pose.position.x
    agent_y = agent_pose.position.y

    # Extract current orientation
    q = [
        agent_pose.orientation.w,
        agent_pose.orientation.x,
        agent_pose.orientation.y,
        agent_pose.orientation.z
    ]
    _, _, yaw = quat2euler(q, 'sxyz')
    return np.array([agent_x, agent_y]), yaw

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

from rclpy.parameter import Parameter
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
# class VelocityListener(Node):
#     def __init__(self):
#         super().__init__('velocity_listener')

#         self.linear_velocity = 0.0

#         self.sub = self.create_subscription(
#             Odometry,
#             '/odometry/filtered',
#             self.callback,
#             QoSProfile(
#                 depth=1,
#                 history=QoSHistoryPolicy.KEEP_LAST,
#                 reliability=QoSReliabilityPolicy.RELIABLE,
#                 durability=QoSDurabilityPolicy.VOLATILE
#                 )
#         )

#     def callback(self, msg):
#         print("call?")
#         self.linear_velocity = msg.twist.twist.linear.x
#         # print(f"Linear velocity: {self.linear_velocity:.3f}")

def main():
    print("entered main")
    parser = argparse.ArgumentParser(
        description='Turtlebot4 Navigation with PID',
    )
    parser.add_argument("--pid-config")
    parser.add_argument("--world")

    args = parser.parse_args()
    print("args parsed")
    with open(args.pid_config, 'r') as file:
    # Use safe_load to convert the YAML content to a Python dictionary
        pid_config = yaml.safe_load(file)

    print("loaded_config")

    rclpy.init()

    env=gym.make('Turtlebot4Env-v0')
    print("env made")
#     node = VelocityListener()

    start_pos, goal_pos, path, is_loop = get_path(pid_config["path"], pid_config["slope"])

    kp = pid_config["kp"]
    ki = pid_config["ki"]
    kd = pid_config["kd"]
    speed = pid_config["speed"]
    horizon = pid_config["horizon"]

#     env.reset()
    state = env.reset(options={"start_pos": start_pos, "goal_pos": goal_pos})
    print("env reset")

    xy, yaw = get_robot_xy(env.env.env.env)
    print(xy, yaw)
        
    xy_err = [0]
    xy_prev = 0

    speed_err = [0]
    speed_prev=0
    done = False
    x_list = [xy[0]]
    y_list = [xy[1]]
    yaw_list = [yaw]
    speed_list = [0]

    dist_err = [0]
    rewards = [0]
    observation, reward, terminated, truncated, info = env.step(np.array([1, 0]))

    curr_index = 0
    i = 0
    while (not terminated):
        # print(i)
        # rclpy.spin_once(node, timeout_sec=0.1)
        
        xy, yaw = get_robot_xy(env.env.env.env)

        closest_path_i, dist = get_closest_index(path, xy[0], xy[1], curr_index, is_loop)

        # dist = np.linalg.norm(path[:, closest_path_i] - np.array([[xy[0]], [xy[1]]]))
        dist_err.append(dist)

        horizon_xy = get_horizon_xy(path, closest_path_i, horizon)

        (dx, dy) = (horizon_xy - xy)

        diff = np.arctan2(dy, dx)
        
        curr_xy_err = diff - yaw

        # make sure angle is in the possible range of angles
        if curr_xy_err > np.pi:
            curr_xy_err = curr_xy_err - 2*np.pi
        elif curr_xy_err < -np.pi:
            curr_xy_err = curr_xy_err + 2*np.pi

        prev_xy_err = xy_err[-1]
        xy_err.append(curr_xy_err)
        
        prev_speed_err = speed_err[-1]
        curr_speed = env.env.env.env.sensors._odom_msg.twist.twist.linear.x
        curr_speed_err = speed - curr_speed
        speed_err.append(curr_speed_err)


        steer = kp[0] * curr_xy_err + ki[0] * sum(xy_err) * 0.05 + kd[0] * (curr_xy_err - prev_xy_err) / 0.05
        # thrust = kp[1] * (curr_speed_err) + ki[1] * sum(speed_err) * 0.05 + kd[1] * (curr_speed_err-prev_speed_err) /0.05
        # print(steer, thrust)
        # observation, reward, terminated, truncated, info = env.step(np.array([steer, thrust]))
        observation, reward, terminated, truncated, info = env.step(np.array([speed, steer]))


        rewards.append(reward)
        x_list.append(xy[0])
        y_list.append(xy[1])
        yaw_list.append(yaw)
        speed_list.append(curr_speed)
        i = i+1

        if xy[0] - 5 > goal_pos[0] or dist > 1:
            terminated=True
            print("Failed")

    df = pd.DataFrame({"x": x_list, "y": y_list, "yaw": yaw_list, "speed": speed_list, "xy_err": dist_err, "yaw_err": xy_err, "speed_err": speed_err, "rewards": rewards})
    df.to_csv(f"{pid_config["path"]}.csv")
    
    print("done")
    env.close()


if __name__ == '__main__':
    main()