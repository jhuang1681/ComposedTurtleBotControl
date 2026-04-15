import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import rclpy

import tb4_drl_navigation.envs # noqa: F401
import yaml
from transforms3d.euler import quat2euler, euler2quat
from get_path import get_path
import time

# The closest index and horizon logic is based off of the race car simulator from assignment 2
# https://github.com/ucsdarclab/RaceCar
def get_closest_index(path: np.ndarray, x: float, y: float, start_ind: float):
    dist_arr = np.sqrt((path[0, :] - x)**2 + (path[1, :]-y)**2)
    return dist_arr.argmin(), dist_arr.min()

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

def main():
    print("entered main")
    parser = argparse.ArgumentParser(
        description='Turtlebot4 Navigation with PID',
    )
    parser.add_argument("--pid-config")
    parser.add_argument("--world")
    parser.add_argument("--load-path", type=str, default=None)
    args = parser.parse_args()
    print("args parsed")
    with open(args.pid_config, 'r') as file:
    # Use safe_load to convert the YAML content to a Python dictionary
        pid_config = yaml.safe_load(file)

    print("loaded_config")

    rclpy.init()
    print("rclpy init")

    world = args.world

    env=gym.make('Turtlebot4Env-v0', world_name=world, map_path=Path(f"/workspaces/cs558_proj/maps/{world}.pgm"), yaml_path=Path(f"/workspaces/cs558_proj/maps/{world}.yaml"), shuffle_on_reset=False)
    print("env made")

    start_pos, goal_pos, path = get_path(pid_config["path"], args.load_path)

    kp = pid_config["kp"]
    ki = pid_config["ki"]
    kd = pid_config["kd"]
    speed = pid_config["speed"]
    horizon = pid_config["horizon"]

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
    start = time.time()
    observation, reward, terminated, truncated, info = env.step(np.array([1, 0]))

    curr_index = 0
    i = 0
    prev_cte = 0.0
    cte_err = [0]
    actual_ctes = [0]
    while (not terminated):
        xy, yaw = get_robot_xy(env.env.env.env)

        closest_path_i, dist = get_closest_index(path, xy[0], xy[1], curr_index)

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

        cte = -np.sin(yaw) * dx + np.cos(yaw) * dy
        cte_err.append(cte-prev_cte)

        (dx_0, dy_0) = get_horizon_xy(path, closest_path_i,0)
        actual_cte = -np.sin(yaw) * dx_0 + np.cos(yaw) * dy_0
        actual_ctes.append(actual_cte)
        # steer = kp[0] * curr_xy_err + ki[0] * sum(xy_err) * 0.05 + kd[0] * (curr_xy_err - prev_xy_err) / 0.05
        steer_yaw = kp[0] * curr_xy_err + ki[0] * sum(xy_err) * 0.05 + kd[0] * (curr_xy_err - prev_xy_err) / 0.05
        steer_cte =  np.arctan2(kp[1] * cte, curr_speed) +  ki[1] * sum(cte_err) + (cte - prev_cte) * kd[1]
        prev_cte = cte
        # steer = kp[0] * curr_xy_err + ki[0] * sum(xy_err) * 0.05 + kd[0] * (curr_xy_err - prev_xy_err) / 0.05 + np.arctan2(kp[1] * cte, speed)
        # steer_cte=0
        steer = steer_yaw + steer_cte
        print(speed, steer_yaw, steer_cte, steer)# steer = kp[0] * curr_xy_err + ki[0] * sum(xy_err) * 0.05 + kd[0] * (curr_xy_err - prev_xy_err) / 0.05 + 0.1 * cte

        # steer = alpha * steer
        steer = np.clip(steer, -np.pi/2, np.pi/2)
        observation, reward, terminated, truncated, info = env.step(np.array([speed, steer]))

        rewards.append(reward)
        x_list.append(xy[0])
        y_list.append(xy[1])
        yaw_list.append(yaw)
        speed_list.append(curr_speed)
        i = i+1

        if xy[0] - 5 > goal_pos[0]:
            terminated=True
            print("Failed")

    df = pd.DataFrame({"x": x_list, "y": y_list, "yaw": yaw_list, "speed": speed_list, "xy_err": dist_err, "yaw_err": xy_err, "speed_err": speed_err, "rewards": rewards, "cte_err": cte_err, "cte": actual_ctes})
    df.to_csv(f"{pid_config["path"]}_{args.load_path}.csv")
    print("done")
    env.close()
    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()