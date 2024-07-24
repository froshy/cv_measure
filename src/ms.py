from classImageInfo import ImageInfo
from configuration import *
import func_utils
import numpy as np
import cv2 as cv
import os 

def measure_stent(img:ImageInfo):
    print(f'On image {os.path.basename(img.img_path)}')
    scan_boxes = img.get_scan_boxes()
    axes = img.get_axes()
    midpoints = img.get_midpoints()
    contoured_img = img.get_contoured_image(image=img.get_morph_image())
    lines = []
    count = 0
    for box, ax, mdpt in zip(scan_boxes, axes, midpoints):
        count += 1
        if count != 4:
            continue
        
        print(f'box: {box}')
        print(f'axis: {ax}')
        print(f'mdpt: {mdpt}')
        box_width, box_height, neg_side, pos_side = func_utils.width_height(box, ax, mdpt)
        n1, n2 = neg_side                                                   # by the way we constructed the boxes, there should always be 2 values in neg side and 2 in pos side
        p1, p2 = pos_side
        vec_n = (n2-n1)/np.linalg.norm(n2-n1)                           # n1 is our starting point
        vec_p = (p2-p1)/np.linalg.norm(p2-p1)
        dist_bt_st = box_height / (SCAN_FREQ + 1)
        pt_iter_n = n1.copy()
        pt_iter_p = p1.copy()
        start_neg = []
        start_pos = []
        for i in range(1,SCAN_FREQ+1):                                  # start at 1 so we aren't scanning edge of box, and +2 to divide box correctly
            start_neg.append(pt_iter_n + vec_n*dist_bt_st)
            start_pos.append(pt_iter_p + vec_p*dist_bt_st)
            pt_iter_n = pt_iter_n + vec_n*dist_bt_st
            pt_iter_p = pt_iter_p + vec_p*dist_bt_st
        scan_direction = np.array([-1,1]) * np.flip(ax, axis=0)
        scan_direction = scan_direction / np.linalg.norm(scan_direction)
        for (pt_neg), (pt_pos) in zip(start_neg, start_pos):
            wt_px_n = _white_pixel_coords(contoured_img, pt_neg, scan_direction, mdpt)
            wt_px_p = _white_pixel_coords(contoured_img, pt_pos, scan_direction, mdpt)
            #line = [wt_px_n, wt_px_p]
            #lines.append(line)
            lines.append([pt_neg, wt_px_n])
            lines.append([pt_pos, wt_px_p])
    return lines
        
def _white_pixel_coords(img, pt, scan_dir, ref_pt):
    x,y = pt
    if np.linalg.norm((x,y) - 2*scan_dir - ref_pt) >= np.linalg.norm((x,y) + 2*scan_dir - ref_pt):
        direction = scan_dir
    else:
        direction = -1 * scan_dir
        
    count = 0
    #is_black(img, np.round(x).astype(int), np.round(y).astype(int)) and
    while is_black(img, np.round(x).astype(int), np.round(y).astype(int)) and count < 30:
        count += 1
        x, y = (x,y) + direction
        if x < 0 or np.round(x) >= img.shape[0] or y < 0 or np.round(y) >= img.shape[1]:
            x,y = pt
            direction *= -1
    white_coord = [x,y]
    print(f'wt cd: {white_coord}')
    return white_coord
        # if np.linalg.norm(n1 - 2*scan_direction - mdpt) >= np.linalg.norm(n1 + 2*scan_direction - mdpt):
        #     for (x_n, y_n), (x_p, y_p) in zip(start_neg, start_pos):
        #         count_1 = 0
        #         count_2 = 0
        #         while is_black(contoured_img, round(x_n), round(y_n)):
        #             print(f'x_n: {x_n}')
        #             print(f'y_n: {y_n}')
        #             x_n, y_n = (x_n,y_n) + scan_direction
        #             count_1 += 1
        #         white_coord_n = [x_n, y_n]
                
        #         while is_black(contoured_img, round(x_p), round(y_p)) and count_2 < 60:
        #         #for _ in range(200):
        #             x_p, y_p = (x_p, y_p) - scan_direction
        #             count_2 += 1
        #         white_coord_p = [x_p, y_p]
                
                
    #     else:
    #         continue
    #         for (x_n, y_n), (x_p, y_p) in zip(start_neg, start_pos):
    #             while is_black(contoured_img, round(x_n), round(y_n)):
    #                 x_n, y_n = (x_n,y_n) - scan_direction
    #             white_coord_n = [x_n, y_n]
    #             while is_black(contoured_img, round(x_p), round(y_p)):
    #                 x_p, y_p = (x_n, y_n) + scan_direction
    #             white_coord_p = [x_p, y_p]
    #     line = [white_coord_n, white_coord_p]
    #     lines.append(line)
    # return lines
        
                
        
        
        
        
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
            
            