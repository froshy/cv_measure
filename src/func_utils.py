import numpy as np

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