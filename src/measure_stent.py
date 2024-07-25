from classImageInfo import ImageInfo
from configuration import *
import func_utils
import numpy as np
import cv2 as cv
import os 


def measure_stent(img:ImageInfo, contoured_img):
    scan_boxes = img.get_scan_boxes()
    axes = img.get_axes()
    midpoints = img.get_midpoints()
    lines = []
    min_dists = []
    mdpts = []
    for box, ax, mdpt in zip(scan_boxes, axes, midpoints):
        min_dist = np.inf
        _, box_height, neg_side, pos_side = func_utils.width_height(box, ax, mdpt)
        n1, n2 = neg_side                                                   # by the way we constructed the boxes, there should always be 2 values in neg side and 2 in pos side
        p1, p2 = pos_side
        vec_n = _get_unit_vec(n1, n2)
        vec_p = _get_unit_vec(p1, p2)
        start_neg = _get_starting_pts(n1, vec_n, box_height, SCAN_FREQ)
        start_pos = _get_starting_pts(p1, vec_p, box_height, SCAN_FREQ)
        scan_direction = np.array([-1,1]) * np.flip(ax, axis=0)
        scan_direction = scan_direction / np.linalg.norm(scan_direction)
        
        for (pt_neg), (pt_pos) in zip(start_neg, start_pos):
            wt_px_n = _white_pixel_coords(contoured_img, pt_neg, scan_direction, mdpt)
            wt_px_p = _white_pixel_coords(contoured_img, pt_pos, scan_direction, mdpt)
            dist = np.linalg.norm(wt_px_n - wt_px_p)
            if dist < min_dist:
                min_dist = dist
                line = [wt_px_n, wt_px_p]
        lines.append(line)
        min_dists.append(min_dist)
        mdpts.append(mdpt)
    return np.array(lines), np.array(min_dists), np.array(mdpts)

###

# Use one or other in single channel images. The images these are being called on should be binary (every pixel is either black (0) or white (255))
    
def is_black(img, y, x):
    if img[x,y] < BLACK_PIXEL_THRESHOLD:
        return True
    return False

def is_white(img, x, y):
    if img[x,y] > WHITE_PIXEL_THRESHOLD:
        return True
    return False

### 

def px_to_len_rate(img_info:ImageInfo, ref_index):
    box = img_info.get_boxes()[ref_index]
    ax = img_info.get_axes()[ref_index]
    mdpt = img_info.get_midpoints()[ref_index]
    
    _, height, _, _ = func_utils.width_height(box, ax, mdpt)
    return REFERENCE_OBJECT_LENGTH / height


def _get_starting_pts(st, direction, tot_dist, sc_freq):
    start_arr = []
    dist_bt_pt = tot_dist / (sc_freq + 1)
    for _ in range(1, sc_freq+1):
        start_arr.append(st + direction*dist_bt_pt)
        st = st + direction*dist_bt_pt
    return start_arr

def _get_unit_vec(st, end):
    return (end-st)/np.linalg.norm(end-st)

def _white_pixel_coords(img, pt, scan_dir, ref_pt):
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
        

            
            