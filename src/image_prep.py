import cv2
import glob
import imutils
import numpy as np
from imutils import perspective

def load_images(img_dir, ext=""):
    """
    Returns a list of images read by cv2.imread()

    Args:
        img_dir (pathlib.Path): path to image directory
        
        ext (str): indicate extension of images e.g. .jpg, .png, etc.
                   If empty string "", attempts to call cv2.imread() on all files in img_dir
                   Defaults to "".
    Returns:
        list: a python array containing the images read by cv2.imread()
    """
    res = []
    for file in glob.glob(img_dir.name + f'/*{ext}'):
        res.append(cv2.imread(file))
    return res


def morph_image(img, canny_th1, canny_th2):
    """
    Processes the images. Blurs image using Gaussian Blur then calls Canny edge algorithm on image, 
    Then dilates and erodes image.
    CHANGES IMAGES TO 1 CHANNEL GRAYSCALE

    Args:
        img (np.array) : matrix representation of image of interest
        canny_th1 (int): threshold1 value
        canny_th2 (int): threshold2 value

    Returns:
        (np.array): returns the processed image
    """
    res_img = cv2.GaussianBlur(img, (5,5), 0)

    res_img = cv2.Canny(res_img, canny_th1, canny_th2)
    res_img = cv2.dilate(res_img, None, iterations=5)
    res_img = cv2.erode(res_img, None, iterations=5)
    
    return res_img

def find_contours(proc_img, fill_contours=True):
    """
    Finds the contours given that it has been morphologically transformed.

    Args:
        proc_img (np.array): an image, typically has been morphologically transformed
        
        fill_contours (bool, optional): Fills in gaps within contours when drawn. MODIFIES (proc_img)
                                        Defaults to True.

    Returns:
        _type_: _description_
    """
    cnts = cv2.findContours(proc_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if fill_contours:
        cv2.drawContours(proc_img, cnts, -1, (255,255,255), cv2.FILLED)
    
    return cnts

def get_boxes(cnts, contour_threshold):
    """
    Returns the minimum area box around the contours cnts

    Args:
        cnts ([int]): list of contours generated from imutils.grab_contours
        contour_threshold (int): minimum area of contour to be boxed

    Returns:
        _type_: _description_
    """
    res_boxes = []
    
    
    for c in cnts:
        if cv2.contourArea(c) < contour_threshold:
            continue
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        res_boxes.append(box)
        
    return res_boxes

def draw_box(img, box):
    img_copy = img.copy()
    for b in box:
        cv2.drawContours(img_copy, [b.astype("int")], -1, (255,255,255), 1)
    return img_copy

def line_along_axis(center, ax, length):
    """
    Gives a line with its midpoint at (center) in the direction of (ax) with length (length)
    Returns a line described by its 2 endpoints

    Args:
        center (numpy array): the center point of the line wanted, e.g. [1,1]
        ax (numpy array: the direction of the line wanted
        length (int): the length of the line wanted

    Returns:
        float, float: returns a line described by its 2 endpoints
    """
    end_pt1 = center + (length/2)*ax[0]
    end_pt2 = center - (length/2)*ax[0]
    end_pt1 = np.squeeze(end_pt1, 0)
    end_pt2 = np.squeeze(end_pt2, 0)
    return end_pt1.astype(int), end_pt2.astype(int)