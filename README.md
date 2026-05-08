# Workspace setup

We utilized a ROS2 workspace template to set up our Gazebo simulation. 

You must have Docker, VSCode, and the corresponding extensions within your VSCode environment.

Clone our repo/download the code.

VSCode should prompt you to open it in a dev container, and if not, do so manually.

For more detailed steps or FAQ related to this workspace setup, please see the README of [VSCode ROS2 Workspace](https://github.com/athackst/vscode_ros2_workspace).

# Gazebo Turtlebot Environment setup

The following steps can also be viewed in the README of [gym-turtlebot](https://github.com/anurye/gym-turtlebot)

To install theTurtlebot simulator

1. Clone [gym-turtlebot](https://github.com/anurye/gym-turtlebot) into workspace.
2. Run the setup commands:
   ```bash
	./setup.sh
	./build.sh

	source install/local_setup.bash
	```
    ** These make ros2 available in the terminal and must be run prior to running any of the simulation or pid control steps.

We made a slight modification to gym-turtlebot in order to be able to get the speed from the odometry sensor.

In gym-turtlebot/src/tb4_drl_navigation/tb4_drl_navigation/envs/utils/ros_gz.py on line 104, there is the odom_callback function.
Add the following line to that function:
```
self._odom_msg = msg
```

# Testing the Simulator

Within a terminal in your ROS2 workspace, run the setup commands specified in step 2 of Gazebo Turtlebot Environment setup.

You may need to enable Docker to be able to use a GUI. On a Windows machine, this may require an installation of the software VcXsrv. Launch this software with disabled access control.

Then, within your docker container run:
```
export DISPLAY=host.docker.internal:0
ros2 launch tb4_gz_sim simulation.launch.py
```

This is a default world, if the environment spawns, then the Gazebo simulator works. You must run the simulator with the default world to ensure proper functioning prior to running PID control.


# Running PID control

** Set up a python virtual environment. We have included a requirements.txt for reference within our code as well.

Within one terminal, ensure the simulator runs using the above Testing the Simulator Instructions.

Run the following simulator command
```
ros2 launch tb4_gz_sim simulation.launch.py world:=<path_to_world_sdf>
```

We have the different worlds of uphill, flat, and downhill in the worlds folder, but for milestone 2 we decided to focus on only a flat world since the slopes did not make a significant difference.

If you would like to run with a slope, choose the path in the config that you would like to use. Ensure the slope corresponds to the world you chose (uphill is -0.1745, downhill is 0.1745, and flat is 0).

In a separate terminal then that in which the simulator is running, within your python environment, run all the Gazebo Turtlebot Environment setup commands. 

Then, run the pid_controller as follows:
```
python pid_controller.py --pid-config <path_to_line_config>
```

# Running HRL \(DQN) Training 
Run the following simulator command:
```
ros2 launch tb4_gz_sim simulation.launch.py headless:=True rviz:=False world:=worlds/flat.sdf
```
This runs the simulation headless and without visualizations.

In a separate terminal, activate and setup your virtual environment and run the following
```
python reinforcement.py --pid-configs <path to each pid config file> --world flat --iter <iteration_num> --episodes <episode_num> --k 5 --max-steps <max_step_num> --update-target 800
```
Additionally, to run the training off a previously trained model, you can add the following argument
```
--checkpoint <path to model>
```

The training saves a model and a csv file of rewards through the episodes.

The rewards of the model training can be plotted using `plot_rewards.py`

# Testing 
To visualize how the robot follows a trajectory, either run `pid_controller.py` with a given pid config or `test_reinforcement.py` with a model. Both will return a csv file of position. These can be plotted using `plot_path.py`.



