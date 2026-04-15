import numpy as np
import matplotlib.pyplot as plt
track_len = 5000

def get_path(path_type: str, load_path: str = None):
    start_pos = np.array([0, 0, 0], dtype=np.float64) # (x, y, yaw)
    goal_pos = np.array([10*np.pi], dtype=np.float64)
    t = np.linspace(0, 10*np.pi, track_len)

    if (load_path is not None):
        loaded_path = np.loadtxt(load_path)
        print("loaded path")
        return start_pos, np.array([loaded_path[0, -1], loaded_path[1, -1], 0]), loaded_path
        

    if path_type == "random":
        start_pos, goal_pos, path = generate_path(12, 0.05, 0.1)
    elif path_type == "line":
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

    # return start_pos, goal_pos, path, is_loop
    return start_pos, goal_pos, path

def generate_path(num_segments, min_len_frac, max_len_frac):
    segments = []
    curr_x = 0.0
    curr_y = 0.0
    curr_heading = 0.0
    raw_pts = np.random.uniform(min_len_frac, max_len_frac, num_segments)
    normalized_pts = raw_pts / raw_pts.sum()
    seg_pts = (normalized_pts * track_len).astype(int)
    seg_pts[np.argmax(seg_pts)] += track_len - seg_pts.sum()
    for i, num_seg_pts in enumerate(seg_pts):
        if i % 2 == 0:
            seg_type = 'line'
        else: seg_type = 'mid_sin'
        # seg_type = np.random.choice(['line', 'mid_sin'])
        # print(seg_type)
        if seg_type == 'line':
            length = np.random.uniform(1.0 , 3.0) # TODO: check straight line length
            local_x = np.linspace(0, length, num_seg_pts)
            local_y = np.zeros_like(local_x)
            end_heading_local = 0.0
        else:
            length = np.random.uniform(0.5, 5.0) # TODO: check sine_length
            sign = np.random.choice([-1, 1])
            local_x = np.linspace(0, length, num_seg_pts)
            local_y = sign * np.sin(3 * local_x) / 3  
            end_heading_local = 0.0

        cos_h = np.cos(curr_heading)
        sin_h = np.sin(curr_heading)
        world_x = curr_x + cos_h * local_x - sin_h * local_y
        world_y = curr_y + sin_h * local_x + cos_h * local_y
        segments.append(np.vstack([world_x, world_y]))

        # Update state for next segment
        curr_x = world_x[-1]
        curr_y = world_y[-1]
        curr_heading = curr_heading + end_heading_local

    path = np.hstack(segments)
    return np.array([path[0,0], path[0,1], 0.0]), np.array([path[-1, 0], path[-1, 1], 0.0]), path

if __name__ == '__main__':
    start, goal, path = get_path("random")
    plt.figure()
    plt.plot(path[0], path[1])
    plt.plot(path[0], path[1])
    plt.gca().set_aspect('equal', adjustable='datalim')  # expands datalim, not box
    plt.grid()
    plt.title("Random piecewise path")
    plt.show()
    plt.savefig("path_check.png")
