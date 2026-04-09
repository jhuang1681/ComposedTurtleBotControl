import numpy as np

track_len = 5000

def get_path(path_type: str):
    start_pos = np.array([0, 0, 0], dtype=np.float64) # (x, y, yaw)
    goal_pos = np.array([10*np.pi], dtype=np.float64)
    t = np.linspace(0, 10*np.pi, track_len)
    if path_type == "line":
        path = np.vstack([t, np.zeros(track_len)])

    elif path_type == "loose_sin":
        t = np.linspace(0, 10 * np.pi, track_len)
        path = np.vstack([t, np.sin(t) / 3])

    elif path_type == "mid_sin":
        t = np.linspace(0, 10 * np.pi, track_len)
        path = np.vstack([t, np.sin(3*t) / 3])
        
    elif path_type == "tight_sin":
        t = np.linspace(0, 10 * np.pi, track_len)
        path = np.vstack([t, np.sin(5*t) / 3])
        
    # elif path_type == "circle": # not sure
    #     t = np.linspace(-1 / 2 * np.pi, 3 / 2 * np.pi, track_len)
    #     path = 10 * np.vstack([np.cos(t), np.sin(t) + 1])
    #     goal_pos = np.array([0, 0, 0], dtype=np.float64)
    #     is_loop=True
    # elif path_type == "line_par_slope":
    #     start_pos = np.array([0, 0, np.pi/2], dtype=np.float64) # (x, y, yaw)

    #     t = np.linspace(0, 50, track_len)
    #     path = np.vstack([
    #             0 * np.ones(track_len),
    #             t,
    #         ])
    #     goal_pos = np.array([0, 50, 0], dtype=np.float64)
    #     is_loop=False
        
    # return start_pos, goal_pos, path, is_loop
    return start_pos, goal_pos, path