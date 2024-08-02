import cv2 as cv
import imutils
from imutils import perspective
import numpy as np

from configuration import *
import func_utils

class ImageInfo:
    
    
    def __init__(self, img):
        """
        Initializes an ImageInfo object

        Args:
            img (2d array): an image in matrix form, typically from cv.imread(image)
        
        self.image (2d array): the image as a matrix
        self.contours (???): the contours in the image
        self.boxes (???): the min-area rectangles around each contour of size greater than specified, in order of left->right in image
        self.midpoints (???): the mid points of each detected object in image, in same order as self.boxes
        self.axes(???): the orientation of each box, direction along longer dimension, in same order as self.boxes
        self.scan_boxes (???): small box to scan for black/white edge detection, in same order as self.boxes
        
        """
        self.image = img            # may have to edit when capturing images from camera
        self.contours = self._initialize_contours()
        self.boxes = self._initialize_boxes()
        
        midpts, axs = self._initialize_midpoint_axis()
        self.midpoints = midpts
        self.axes = axs
        self.px2len_rate = self._initialize_px2len_rate()
        self.scan_boxes = self._initialize_scan_boxes()
        
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
        lower = int(max(0, (1-CANNY_SIGMA) * v))
        upper = int(min(255, (1+CANNY_SIGMA) * v))
        #morph_img = cv.GaussianBlur(self.get_image(), (5,5), 1)
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
        """
        Initializes scan boxes. 

        Returns:
            ([np.array(ints)]): a list of scan boxes
        """
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
        """
        Checks the corners of a box to make sure the coordinates are within the image. 
        If corners aren't in the image, then they are moved to the edge of the image, preserving orientation.

        Args:
            corners (???): list of corner coordinates to check
            index (int): index of box we are checking (index refers to the box's corresponding order in class)

        Returns:
            ???: Returns corner coordinates, adjusted if original coordinates outside of image
        """
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
        box = self.get_boxes()[REFERENCE_INDEX]
        ax = self.get_axes()[REFERENCE_INDEX]
        mdpt = self.get_midpoints()[REFERENCE_INDEX]
        
        _, height, _, _ = func_utils.width_height(box, ax, mdpt)
        return REFERENCE_OBJECT_LENGTH / height
    
    def measure_objects(self):
        scan_boxes = self.get_scan_boxes()
        axes = self.get_axes()
        midpoints = self.get_midpoints()
        lines = []
        min_dists = []
        mdpts = []
        contoured_img = self.get_contoured_image(self.get_morph_image())
        for box, ax, mdpt in zip(scan_boxes, axes, midpoints):
            min_dist = np.inf
            _, box_height, neg_side, pos_side = func_utils.width_height(box, ax, mdpt)
            n1, n2 = neg_side                                                   # by the way we constructed the boxes, there should always be 2 values in neg side and 2 in pos side
            p1, p2 = pos_side
            vec_n = func_utils.get_unit_vec(n1, n2)
            vec_p = func_utils.get_unit_vec(p1, p2)
            start_neg = func_utils.get_starting_pts(n1, vec_n, box_height, SCAN_FREQ)
            start_pos = func_utils.get_starting_pts(p1, vec_p, box_height, SCAN_FREQ)
            scan_direction = np.array([-1,1]) * np.flip(ax, axis=0)
            scan_direction = scan_direction / np.linalg.norm(scan_direction)
            
            for (pt_neg), (pt_pos) in zip(start_neg, start_pos):
                wt_px_n = func_utils.white_pixel_coords(contoured_img, pt_neg, scan_direction, mdpt)
                wt_px_p = func_utils.white_pixel_coords(contoured_img, pt_pos, scan_direction, mdpt)
                dist = np.linalg.norm(wt_px_n - wt_px_p)
                if dist < min_dist:
                    min_dist = dist
                    line = [wt_px_n, wt_px_p]
            lines.append(line)
            min_dists.append(min_dist)
            mdpts.append(mdpt)
        return np.array(lines), np.array(min_dists), np.array(mdpts)
    
    def get_measured_image(self, image=None, in_unit=True, display_line=True):
        lines, min_dists, _ = self.measure_objects()
        if image is None:
            image = self.get_image()
        if in_unit:
            min_dists = min_dists * self.px2len_rate
        if display_line:
            proc_img = self.get_line_draw_img(lines, image)
        proc_img = self.get_display_vals_img(image, min_dists)
        return proc_img
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
        """
        displays contoured image. 

        Args:
            image (matrix, optional): image to draw contours on. If None, draws on original image. Defaults to None.
            contour_color (tuple, optional): Color is BGR to draw contours with. Defaults to (255,255,255).
            fill_contour (bool, optional): Option to fill in contour or not. Defaults to True.
        """
        if image is None:
            image = self.image.copy()
        contoured_image = self.get_contoured_image(image=image, contour_color=contour_color, fill_contour=fill_contour)
        cv.imshow(f'contoured', contoured_image)
    
    def get_scan_box_image(self, image=None, box_color=(255,255,255), box_id=-1, box_thickness =1):
        """
        Returns image with scan boxes drawn on

        Args:
            image (matrix, optional): the image to draw on. If None, uses original image. Defaults to None.
            box_color (tuple, optional): Color in BGR to draw boxes with. Defaults to (255,255,255).
            box_id (int, optional): The index of the box to draw. If -1, draws all boxes. Defaults to -1.

        Returns:
            matrix: image with scan boxes drawn on
        """
        if image is None:
            image = self.image.copy()
        else:
            image = image.copy()
        cv.drawContours(image, np.array(self.scan_boxes), box_id, box_color, box_thickness)
        return image
    
    def get_boxed_image(self, image=None, box_color=(255, 255, 255), box_id=-1, box_thickness=1):
        """
        Returns image with boxes drawn on

        Args:
            image (matrix, optional): the image to draw on. If None, uses original image. Defaults to None.
            box_color (tuple, optional): Color in BGR to draw boxes with. Defaults to (255,255,255).
            box_id (int, optional): The index of the box to draw. If -1, draws all boxes. Defaults to -1.
            box_thickness(int, optional): How thick to draw lines. Defaults to 1
        Returns:
            matrix: image with scan boxes drawn on
        """
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
            matrix: image with contours drawn on
        """
        if image is None:
            image = self.image.copy()
        if fill_contour:
            cv.drawContours(image, self.contours, -1, contour_color, cv.FILLED)
        else:
            cv.drawContours(image, self.contours, -1, contour_color, 1)
        return image
    
    def get_line_draw_img(self, lines, image=None, line_color=(0,0,255)):
        """
        Returns image with lines draw on

        Args:
            lines ([[int,int],[int,int]]): list of pairs of coordinates indicating the endpoints of a lien
            image (matrix, optional): image as a matrix. Defaults to None.
            line_color (tuple, optional): color is BGR to make the line. Defaults to (0,0,255).

        Returns:
            matrix: image
        """
        if image is None:
            image = self.get_image()
        else:
            image = image.copy()
        for st, end in lines:
            cv.line(image, np.round(st).astype(int), np.round(end).astype(int), line_color, 1)
        return image
    
    def get_display_vals_img(self, image, vals, round=5):
        """
        Returns image with values drawn on

        Args:
            image (matrix): an image as a matrix
            vals (anything that's string convertible): list of values, should equal the number of objects, 
                                                        values should be in order with corresponding objects

        Returns:
            matrix: image
        """
        assert len(vals) == len(self.get_axes())
        vals = np.round(vals, round)
        font_scale = self.get_image().shape[0] * self.get_image().shape[1] / (1920 * 1080)
        img_height = self.get_image().shape[0]
        img_width = self.get_image().shape[1]
        fontScale = min(img_width,img_height)/(700)
        thickness = np.max([int(3 * fontScale), 1])
        for val, mdpt in zip(vals, self.get_midpoints()):
            cv.putText(image, str(val), np.round(mdpt).astype(int), cv.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,255), thickness)
        return image
    
    def get_image(self):
        """
        Returns attribute image

        Returns:
            (np.array): a matrix representation of image
        """
        return self.image.copy()
    
    def pix_to_len(self, n_pixels):
        """
        Returns converted value from pixels to unit length

        Args:
            n_pixels (float or int): the number of pixels to convert to length

        Returns:
            float: the converted value
        """
        return self.px2len_rate * n_pixels
    
    def get_scan_boxes(self):
        """
        Returns scan boxes

        Returns:
            (scan box type): the scan boxes, in order from left to right in image
        """
        return self.scan_boxes
    
    def get_boxes(self):
        """
        Returns min_area_rect of objects detected

        Returns:
            (box type): the boxes, in order from left to right in image
        """
        return self.boxes
    
    def get_midpoints(self):
        """
        Returns midpoints of objects detected

        Returns:
            (list of coordinates): list of midpoints of objects detected
        """
        return self.midpoints
    
    def get_axes(self):
        """
        Returns axes (orientation) of objects detected

        Returns:
            (list of [x,y]): list of directions of objects (along longest dimension) in order of from left -> right
        """
        return self.axes
    
    def get_px2len_rate(self):
        return self.px2len_rate