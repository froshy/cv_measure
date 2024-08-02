import numpy as np
from configuration import *
def width_height(pts, ax, mdpt):
    """
    Returns width and height of rectangular object that sits square along axis.
    Must know midpoint
    Width is dimension perpendicular to axis.
    Height is dimension parallel to axis

    Args:
        pts ([[int]]): List of 4 coordinate points e.g [[x1,y1],..., [x4,y4]]
        axis ([float]): Direction of axis e.g. [3,4] represents the vector (3,4)
        
    neg_side and pos_side are points from pts that are separated by ax
    
    Returns width, height, neg_side, pos_side -> (float, float, [[float]], [[float]])
    """
    x_1, y_1 = mdpt + 10 * ax                                           # actual value of the 10 doesn't matter, only that we need 2 points on the axial line that goes through the midpoint
    x_2, y_2 = mdpt - 10 * ax
    neg_side = []
    pos_side = []
    for x,y in pts:
        if np.sign((x-x_1) * (y_2 - y_1) - (y-y_1) * (x_2-x_1)) < 0:    # want to differentiate the two points one one side from the other two points on the other side
            neg_side.append(np.array([x,y]))
        else:
            pos_side.append(np.array([x,y]))
            
    width = np.min([np.linalg.norm((neg_side[0] - pos_side[i])) for i in range(len(pos_side))])
    height = np.linalg.norm(neg_side[0] - neg_side[1])
    return width, height, neg_side, pos_side



############################################################ 
# measure_object() helpers
def get_starting_pts(st, direction, tot_dist, sc_freq):
    start_arr = []
    dist_bt_pt = tot_dist / (sc_freq + 1)
    for _ in range(1, sc_freq+1):
        start_arr.append(st + direction*dist_bt_pt)
        st = st + direction*dist_bt_pt
    return start_arr

def get_unit_vec(st, end):
    return (end-st)/np.linalg.norm(end-st)

def white_pixel_coords(img, pt, scan_dir, ref_pt):
    x,y = pt
    if np.linalg.norm((x,y) - 2*scan_dir - ref_pt) >= np.linalg.norm((x,y) + 2*scan_dir - ref_pt):
        direction = scan_dir
    else:
        direction = -1 * scan_dir
    count = 0
    max_iter = np.sum(img.shape)
    while is_black(img, np.round(x).astype(int), np.round(y).astype(int)) and count <= max_iter:
        count += 1
        x, y = (x,y) + direction
        if x < 0 or np.round(x) >= img.shape[1] or y < 0 or np.round(y) >= img.shape[0]:
            x,y = pt
            direction *= -1
    if count >= max_iter:
        raise Exception("Could not locate object")
    white_coord = np.array([x,y])
    return white_coord

def is_black(img, y, x):
    if img[x,y] <= BLACK_PIXEL_THRESHOLD:
        return True
    return False

def is_white(img, x, y):
    if img[x,y] >= WHITE_PIXEL_THRESHOLD:
        return True
    return False
############################################################ 