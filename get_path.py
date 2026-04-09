import numpy as np

track_len = 5000

def get_path(path_type: str, slope: float): # slope in radians
    start_pos = np.array([0, 0, 0], dtype=np.float64) # (x, y, yaw)
    if path_type == "line":
        t = np.linspace(0, 50, track_len)
        path = np.vstack([
                t * np.cos(slope),
                0 * np.ones(track_len)
            ])
        goal_pos = np.array([50, 0, 0], dtype=np.float64)
        is_loop=False

    elif path_type == "loose_sin":
        t = np.linspace(0, 10 * np.pi, track_len)
        path = np.vstack([
		t * np.cos(slope),
		np.sin(t) / 3
	    ])
        goal_pos = np.array([10 * np.pi, 0, 0])
        is_loop=False
        
    elif path_type == "tight_sin":
        t = np.linspace(0, 10 * np.pi, track_len)
        path = np.vstack([
		t * np.cos(slope),
		5 * np.sin(t)
	    ])
        goal_pos = np.array([10 * np.pi, 0, 0], dtype=np.float64)
        is_loop=False
        
    elif path_type == "circle": # not sure
        t = np.linspace(-1 / 2 * np.pi, 3 / 2 * np.pi, track_len)
        path = 10 * np.vstack([np.cos(t), np.sin(t) + 1])
        goal_pos = np.array([0, 0, 0], dtype=np.float64)
        is_loop=True

    elif path_type == "line_par_slope":
        start_pos = np.array([0, 0, np.pi/2], dtype=np.float64) # (x, y, yaw)

        t = np.linspace(0, 50, track_len)
        path = np.vstack([
                0 * np.ones(track_len),
                t,
            ])
        goal_pos = np.array([0, 50, 0], dtype=np.float64)
        is_loop=False
        
#     elif path_type == "track": 
    	# start_pos = (0,0,0)
        # top length
	# right bend
	# bottom length
	# left bend
        # goal_pos = (0, 0, 0)
	# is_loop=True
        
    return start_pos, goal_pos, path, is_loop