import cv2 as cv
import imutils
from imutils import perspective
import numpy as np
import os

from configuration import *
import func_utils

class ImageInfo:
    
    
    def __init__(self, img):
        self.image = img            # may have to edit when capturing images from camera
        self.contours = self._initialize_contours()
        self.boxes = self._initialize_boxes()
        
        midpts, axs = self._initialize_midpoint_axis()
        self.midpoints = midpts
        self.axes = axs
        
        self.scan_boxes = self._initialize_scan_boxes()
        #self.px2len_rate = self._initialize_px2len_rate()
        
    def _initialize_contours(self):
        """
        Finds the contours from a morphed image

        Returns:
            (np.array): Returns the contours of original image
        """
        morph_img = self.get_morph_image()
        contours  = cv.findContours(morph_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours  = imutils.grab_contours(contours)
        return contours
    
    def get_morph_image(self):
        """
        Performs a series of morphological transformation on self.image.
        img -> Gaussian Blur -> Canny -> dilate -> erode

        Returns:
            np.array: the morphed image in matrix form
        """
        morph_img = cv.cvtColor(self.get_image(), cv.COLOR_BGR2GRAY)
        v =  np.median(morph_img)
        lower = int(max(0, 2*(1-CANNY_SIGMA) * v))
        upper = int(min(255, 2*(1-CANNY_SIGMA) * v))
        #lower, upper = 0,255
        #morph_img = cv.GaussianBlur(self.get_image(), (3,3), 0)
        morph_img = cv.Canny(morph_img, lower, upper)
        morph_img = cv.dilate(morph_img, None, iterations=MORPHOLOGICAL_ITERS)
        morph_img = cv.erode(morph_img, None, iterations=MORPHOLOGICAL_ITERS)
        return morph_img
    
    def _initialize_boxes(self):
        """
        Creates the boxes around the detected objects in self.image, returns in order of left -> right in image.

        Returns:
            [[int]]: returns a list of boxes where boxes is a list of points 
                    describing a 4 sided polygon around detected object, length of return value
                    is the number of detected objects with contour area > configuration.CONTOUR_AREA_THRESHOLD
                    boxes are sorted left to right
        """
        res_boxes = []
        for cnt in self.contours:
            if cv.contourArea(cnt) < CONTOUR_AREA_THRESHOLD:
                continue
            box = cv.minAreaRect(cnt)
            box = cv.boxPoints(box)          
            box = np.array(box, dtype='int')
            box = perspective.order_points(box)
            res_boxes.append(box)
        res_boxes = sorted(res_boxes, key=lambda x: x[0][0]) # sort by top left x coordinate in ascending order
        return res_boxes
    
    def _initialize_midpoint_axis(self):
        """
        Locates the midpoints and axis of orientation of each box in self.image.
        Returned in same order as the boxes

        Returns:
            (midpts, axs)
            
            midpts ([[int]]): list of coordinates (in order of boxes) of the midpoints of each box
            axs ([[int]]): list of axes (in order of boxes) of each box; the orientation of the box
        """
        midpts = []
        axs = []
        for box in self.boxes:
            midpoint, ax = cv.PCACompute(box, np.mean(box, axis=0).reshape(1,-1))
            midpts.append(np.squeeze(midpoint))
            axs.append(ax[0]) # take zeroeth index to get first principal componentl this is our orientation
        return midpts, axs
    
    def _initialize_scan_boxes(self):
        scan_boxes = []
        assert len(self.axes) == len(self.midpoints) == len(self.boxes), f"Initialized incorrectly, number of axes ({len(self.axes)}) should equal number of midpoints ({len(self.midpoints)} and the number of boxes ({len(self.boxes)}))"
        for i in range(len(self.axes)):
            width, height, _, _ = func_utils.width_height(self.boxes[i], self.axes[i], self.midpoints[i])
            corner1 = (self.midpoints[i] + (np.array([-1,1]) * np.flip(self.axes[i], axis=0)*width/2*SCAN_WIDTH_TOL) + (self.axes[i] * SCAN_HEIGHT_PROP * height/2))
            corner2 = (self.midpoints[i] + (np.array([-1,1]) * np.flip(self.axes[i], axis=0)*width/2*SCAN_WIDTH_TOL) - (self.axes[i] * SCAN_HEIGHT_PROP * height/2))
            corner3 = (self.midpoints[i] - (np.array([-1,1]) * np.flip(self.axes[i], axis=0)*width/2*SCAN_WIDTH_TOL) + (self.axes[i] * SCAN_HEIGHT_PROP * height/2))
            corner4 = (self.midpoints[i] - (np.array([-1,1]) * np.flip(self.axes[i], axis=0)*width/2*SCAN_WIDTH_TOL) - (self.axes[i] * SCAN_HEIGHT_PROP * height/2))
            box = perspective.order_points(np.array([corner1, corner2, corner3, corner4])).astype(int)
            box = self._check_corners(box, i)
            scan_boxes.append(box)
        return scan_boxes
    
    def _check_corners(self, corners, index):
        axis = self.axes[index]
        h, w, _ = self.image.shape
        res_corner = []
        for x, y in corners:
            new_x = x
            new_y = y
            if y < 0:
                new_y = 0
                new_x = (new_y/axis[0]) * axis[1] + new_x
            if y >= h:
                new_y = h-1
                new_x = ((new_y - (h-1)) /axis[0]) * axis[1] + new_x
            if new_x < 0:
                new_x = 0
                new_y = (new_x/axis[1]) * axis[0] + new_y
            if new_x >= w:
                new_x = w-1
                new_y = ((new_y - (w-1)) / axis[1]) * axis[0] + new_y
            res_corner.append([new_x,new_y])
        return np.array(res_corner).astype(int)
    def _initialize_px2len_rate(self):
        box = self.boxes[REFERENCE_INDEX]
        ax = self.axes[REFERENCE_INDEX]
        mdpt = self.midpoints[REFERENCE_INDEX]
        _, height, _, _ = func_utils.width_height(box, ax, mdpt)
        len_per_pixel = REFERENCE_OBJECT_LENGTH / height
        return len_per_pixel

    def show_morph_image(self):
        """
        Displays image from morphological transforms
        
        Does not return anything
        """
        morph_image = self.get_morph_image()
        cv.imshow(f'morphed', morph_image)
        
    def show_image(self):
        """
        Displays original image
        
        Does not return anything
        """
        cv.imshow('foo', self.image)
    
    def show_contoured_image(self, image=None, contour_color=(255,255,255), fill_contour=True):
        if image is None:
            image = self.image
        contoured_image = self.get_contoured_image(image=image, contour_color=contour_color, fill_contour=fill_contour)
        cv.imshow(f'contoured', contoured_image)
    
    def get_scan_box_image(self, image=None, box_color=(255,255,255), box_id=-1):
        if image is None:
            image = self.image.copy()
        else:
            image = image.copy()
        cv.drawContours(image, np.array(self.scan_boxes), box_id, box_color, 1)
        return image
    def get_boxed_image(self, image=None, box_color=(255, 255, 255), box_id=-1, box_thickness=1):
        if image is None:
            image = self.image.copy()
        else:
            image = image.copy()
        cv.drawContours(image, np.array(self.boxes).astype(int), box_id, box_color, box_thickness)
        return image
    def get_contoured_image(self, image = None, contour_color=(255,255,255), fill_contour=True):
        """
        Gets the image with contours drawn on original copy unless otherwise specified.

        Args:
            image (): Defaults to None
            contour_color (tuple, optional): _description_. Defaults to (255,255,255).
            fill_contour (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if image is None:
            image = self.image.copy()
        if fill_contour:
            cv.drawContours(image, self.contours, -1, contour_color, cv.FILLED)
        else:
            cv.drawContours(image, self.contours, -1, contour_color, 1)
        return image
    
    def get_line_draw_img(self, lines, image=None, line_color=(0,0,255)):
        if image is None:
            image = self.get_image()
        for st, end in lines:
            cv.line(image, np.round(st).astype(int), np.round(end).astype(int), line_color, 1)
        return image
    
    def get_display_vals_img(self, image, vals):
        assert len(vals) == len(self.get_axes())
        vals = np.round(vals, 5)
        for val, mdpt in zip(vals, self.get_midpoints()):
            cv.putText(image, str(val), np.round(mdpt + [10,10]).astype(int), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        return image
    def get_image(self):
        """
        Returns attribute image

        Returns:
            (np.array): a matrix representation of image
        """
        return self.image
    def pix_to_len(self, n_pixels):
        return self.px2len_rate * n_pixels
    def get_scan_boxes(self):
        return self.scan_boxes
    def get_boxes(self):
        return self.boxes
    def get_midpoints(self):
        return self.midpoints
    def get_axes(self):
        return self.axes