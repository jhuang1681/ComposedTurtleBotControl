import numpy as np
import matplotlib.pyplot as plt
# track_len = 5000

def get_path(path_type: str, load_path: str = None, seg_types: list[str] = None):
    track_len = 5000
    start_pos = np.array([0, 0, 0], dtype=np.float64) # (x, y, yaw)
    goal_pos = np.array([10*np.pi,0, 0], dtype=np.float64)
    t = np.linspace(0, 10*np.pi, track_len)
    if (load_path is not None):
        loaded_path = np.loadtxt(load_path)
        print("loaded path")
        return start_pos, np.array([loaded_path[0, -1], loaded_path[1, -1], 0]), loaded_path, len(loaded_path[0])
        

    if path_type == "random":
        if (seg_types is not None):
            num_segments = len(seg_types) * 6
            track_len = 2500 * len(seg_types)
            min_len_frac = 1 / num_segments - 0.02
            max_len_frac = 1 / num_segments + 0.02
        else:
            seg_types = ['line', 'loose_sin', 'mid_sin', 'tight_sin']
            num_segments = 24
            track_len = 10000
            min_len_frac = 0.02
            max_len_frac = 0.06
        start_pos, goal_pos, path = generate_path(seg_types, track_len, num_segments, min_len_frac, max_len_frac)
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

    return start_pos, goal_pos, path, track_len

def generate_path(seg_types, track_len, num_segments, min_len_frac, max_len_frac):
    print(track_len)
    segments = []
    curr_x = 0.0
    curr_y = 0.0
    curr_heading = 0.0
    raw_pts = np.random.uniform(min_len_frac, max_len_frac, num_segments)
    normalized_pts = raw_pts / raw_pts.sum()
    seg_pts = (normalized_pts * track_len).astype(int)
    seg_pts[np.argmax(seg_pts)] += track_len - seg_pts.sum()
    prev_type = None
    for i, num_seg_pts in enumerate(seg_pts):
        choices = seg_types.copy()
        print(choices)
        if prev_type is not None and len(choices) > 1:
            choices.remove(prev_type)
        seg_type = np.random.choice(choices)
        prev_type = seg_type
        if seg_type == 'line':
            length = np.random.uniform(3.0 , 5.0) # originally 1-3, now 3-5
            local_x = np.linspace(0, length, num_seg_pts)
            local_y = np.zeros_like(local_x)
            end_heading_local = 0.0
        else:
            if seg_type == 'loose_sin':
                length = 5.0
                sign = np.random.choice([-1, 1])
                offset = np.random.uniform(0.0, 0.8)
                local_x = np.linspace(offset, length + offset, num_seg_pts)
                
                local_y = sign * np.sin(local_x) / 3  
                
            elif seg_type == 'mid_sin':
                length = 3.0
                sign = np.random.choice([-1, 1])
                offset = np.random.uniform(0.0, 0.8)
                local_x = np.linspace(offset, length + offset, num_seg_pts)
            
                local_y = sign * np.sin(3 * local_x) / 3  
                
            else:
                length = np.random.uniform(1.0 , 1.5)
                sign = np.random.choice([-1, 1])
                offset = np.random.uniform(0.0, 0.8)
                local_x = np.linspace(offset, length + offset, num_seg_pts)
            
                local_y = sign * np.sin(5 * local_x) / 3  
            
            local_x = np.linspace(0, length, num_seg_pts)
            local_y = local_y - local_y[0]    
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
    return np.array([path[0,0], path[1,0], 0.0]), np.array([path[0, -1], path[1, -1], 0.0]), path


def save_path(path, filename):
    np.savetxt(filename, path)
    print(f"Saved path to {filename}")

if __name__ == '__main__':

    max_start_idx = int((1.0 - 0.2) * 5000) # track_len in get_path.py hardcoded as 5000
    start_idx = np.random.randint(0, max_start_idx)
    goal_idx = start_idx + int(0.2 * 5000) 

    start, goal, path = get_path("random")
    # print(path[:, start_idx], path[:, goal_idx])
    plt.figure()
    plt.plot(path[0], path[1])
    # plt.gca().set_aspect('equal', adjustable='datalim')  # expands datalim, not box
    plt.grid()
    plt.title("Random piecewise path")
    plt.savefig("path_check.png")

    save_path(path, "test_path4.txt")
