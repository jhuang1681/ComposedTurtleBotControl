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
# from reinforcement import compute_reward

# The closest index and horizon logic is based off of the race car simulator from assignment 2
# https://github.com/ucsdarclab/RaceCar
def get_closest_index(path: np.ndarray, x: float, y: float, start_ind: float, is_loop: bool = False):
    best_dist = np.inf

    ind_s = [i for i in range(start_ind)]
    ind_e = [start_ind + i for i in range(path.shape[1]-start_ind)]

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
    return i, best_dist

    dist_arr = np.sqrt((path[0, :] - x)**2 + (path[1, :]-y)**2)
    return dist_arr.argmin(), dist_arr.min()

def get_horizon_xy(path: np.ndarray, current_index: float, horizon: int, is_loop: bool=False):
    max_index = path.shape[1] - 1
    
    if is_loop and current_index + horizon > max_index:
        index =  current_index + horizon - max_index
    elif current_index + horizon > max_index:
        index=max_index
    else:
        index = current_index + horizon
    # print(current_index, index)
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

reward_dict = {"running_total": [], "total": [], "r_cte": [], "r_heading": [], "r_dcte": [], "r_speed": [], "r_goal": []}
def compute_reward(state):
    cte, heading_err, speed, dcte, done_goal = state
    w = [2, 0, 0.05, 1, 0.5]
    r_cte = -w[0] * cte**2 # penalize lateral deviation
    r_heading = -w[1] * heading_err**2 # penalize misalignment

    r_dcte = -w[4] * dcte**2 # penalize growing lateral error

    # cos(heading_err) reduces reward when pointing wrong direction
    r_speed = w[2] * speed * np.cos(heading_err) # reward speed (in right direction)
    # --- Sparse terminal rewards ---
    r_goal = 0 #10.0 if done_goal else 0.0

    reward_dict["total"].append(r_cte + r_heading + r_dcte + r_speed + r_goal)
    reward_dict["r_cte"].append(r_cte)
    reward_dict["r_heading"].append(r_heading)
    reward_dict["r_dcte"].append(r_dcte)
    reward_dict["r_speed"].append(r_speed)
    reward_dict["r_goal"].append(r_goal)
    reward_dict["running_total"].append(sum(reward_dict["total"]))
    # print(f"tot: {r_cte + r_heading + r_dcte + r_speed + r_goal}, cte: {r_cte}, head: {r_heading}, dcte: {r_dcte}, speed: {r_speed}, goal: {r_goal}")
    return r_cte + r_heading + r_dcte + r_speed + r_goal
import rclpy

def main():
    print("entered main")
    parser = argparse.ArgumentParser(
        description='Turtlebot4 Navigation with PID',
    )
    parser.add_argument("--pid-config")
    parser.add_argument("--world", type=str, default="flat")
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--min-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--is-loop", type=bool, default=False)
    args = parser.parse_args()
    print("args parsed")
    with open(args.pid_config, 'r') as file:
    # Use safe_load to convert the YAML content to a Python dictionary
        pid_config = yaml.safe_load(file)

    print("loaded_config")

    rclpy.init()
    print("rclpy init")

    world = args.world

    env=gym.make('Turtlebot4Env-v0', world_name=world, map_path=Path(f"/workspaces/cs558_proj/maps/{world}.pgm"), yaml_path=Path(f"/workspaces/cs558_proj/maps/{world}.yaml"), shuffle_on_reset=False, goal_threshold=0.2)
    print("env made")

    start_pos, goal_pos, path, _ = get_path(pid_config["path"], args.load_path)

    kp = pid_config["kp"]
    ki = pid_config["ki"]
    kd = pid_config["kd"]
    speed = pid_config["speed"]
    horizon = pid_config["horizon"]
    is_loop = args.is_loop
    print(args.min_steps)
    min_steps = 0
    if args.min_steps > 0:
        min_steps = args.min_steps
    max_steps = np.inf
    print(args.max_steps)
    if args.max_steps is not None:
        max_steps = args.max_steps
    
    
    state = env.reset(options={"start_pos": start_pos, "goal_pos": goal_pos})
    print("env reset")

    xy, yaw = get_robot_xy(env.env.env.env)
        
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
    closest_path_i = 0
    while (i < min_steps):
        xy, yaw = get_robot_xy(env.env.env.env)

        closest_path_i, dist = get_closest_index(path, xy[0], xy[1], closest_path_i, is_loop)
        print(path[:, closest_path_i], dist, xy)
        # dist = np.linalg.norm(path[:, closest_path_i] - np.array([[xy[0]], [xy[1]]]))
        dist_err.append(dist)

        horizon_xy = get_horizon_xy(path, closest_path_i, horizon, is_loop)

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

        (dx_0, dy_0) = get_horizon_xy(path, closest_path_i,0, is_loop)
        actual_cte = -np.sin(yaw) * dx_0 + np.cos(yaw) * dy_0
        actual_ctes.append(actual_cte)
        # steer = kp[0] * curr_xy_err + ki[0] * sum(xy_err) * 0.05 + kd[0] * (curr_xy_err - prev_xy_err) / 0.05
        steer_yaw = kp[0] * curr_xy_err + ki[0] * sum(xy_err) * 0.05 + kd[0] * (curr_xy_err - prev_xy_err) / 0.05
        # steer_cte =  np.arctan2(kp[1] * cte, curr_speed) +  ki[1] * sum(cte_err) + (cte - prev_cte) * kd[1]
        # prev_cte = cte
        steer = steer_yaw #+ steer_cte
        # print(speed, steer_yaw, steer_cte, steer)# steer = kp[0] * curr_xy_err + ki[0] * sum(xy_err) * 0.05 + kd[0] * (curr_xy_err - prev_xy_err) / 0.05 + 0.1 * cte

        # steer = alpha * steer
        steer = np.clip(steer, -np.pi/2, np.pi/2)
        observation, reward, terminated, truncated, info = env.step(np.array([speed, steer]))

        dcte = cte - prev_cte
        pid_reward = compute_reward([dist, curr_xy_err, curr_speed, dcte, terminated])
        rewards.append(pid_reward)
        print(sum(rewards))
        x_list.append(xy[0])
        y_list.append(xy[1])
        yaw_list.append(yaw)
        speed_list.append(curr_speed)
        i = i+1

        if dist > 0.5:
            print("failed!")
            break
        # if i == max_steps:
        #     print("max reached")
        #     break
        # if xy[0] - 5 > goal_pos[0]:
        #     terminated=True
        #     print("Failed")

        if i % 50 == 0:
            reward_df = pd.DataFrame(reward_dict)
            reward_df.to_csv("reward_df.csv")

    df = pd.DataFrame({"x": x_list, "y": y_list, "yaw": yaw_list, "speed": speed_list, "xy_err": dist_err, "yaw_err": xy_err, "speed_err": speed_err, "rewards": rewards, "cte_err": cte_err, "cte": actual_ctes})
    df.to_csv(f"{pid_config["path"]}_{args.load_path}.csv")
    print("done")
    env.close()
    end = time.time()
    print(end - start)
    reward_df = pd.DataFrame(reward_dict)
    reward_df.to_csv("reward_df.csv")

if __name__ == '__main__':
    main()