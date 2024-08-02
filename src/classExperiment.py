from configuration import *
from classImageInfo import *
import func_utils
import pandas as pd
import numpy as np
import warnings
import time

class MeasureExperiment():
    
    def __init__(self):
        """
        Initializes MeasureExperiment object
        
        self.num_stent (int): number of stents in experiment, edit in configuration.py
        self.n_total_obj (int): number of total objects, typically num_stent + 1 (reference object)
        self.ref_index (int): index of reference object from left -> right in image, 0-indexed
        self.px2len_rate(float): the len per pixel ratio, used to convert num of pixels to length
        self.mdpts(list of midpoints): a list of midpoints when initializing from frame. Used to keep track of which future objects correpond to objects when initialized
        self.log_by_mdpt (dict): dictionary where key is midpoint and value is a list of lengths
        self.time_log (list): list of times of when we log
        self.start_time (float???): time for when logging starts
        """
        self.num_stents = NUM_STENTS
        self.n_total_obj = N_TOTAL_OBJECTS
        self.ref_index = REFERENCE_INDEX
        self.px2len_rate = None
        self.mdpts = None                   # These midpoints correspond to each object, if we detect less objects in the next frame, 
                                            # the detected objects midpoints will correspond to the midpoint that is closest in the existing one.
                                            # if we detect more objects than we initialized, then the objects farthest away from the midpoints will be ignored
        self.log_by_mdpt = None             # initialized as None, but will be a dictionary where key is the midpoint (tuple) and values are a log of lengths associated with the object of the midpoint
        self.time_log = None
        self.temp_log = None
        self.start_time = None
    def initialize_from_frame(self, img:ImageInfo, include_time = True):
        """
        Initializes experiment from img
        Returns False if the number of objects detected in img is equal to self.n_total_obj and True if initialized
        properly

        Args:
            img (ImageInfo): an ImageInfo object
            include_time (bool, optional): option to include time or not. Defaults to True.

        Returns:
            (bool): If False -> expected objects not equal to objects detected, True if successfully initialized
        """
        if len(img.get_axes()) != self.n_total_obj:
            warnings.warn(f'Detected {len(img.get_axes())} objects but expected {self.n_total_obj}')
            return False
        self.px2len_rate = self._initialize_px2len_rate(img)
        self.mdpts = self._initialize_mdpts(img)
        self.log_by_mdpt, self.time_log, self.temp_log = self._initialize_logs(include_time)
        self.start_time = time.time()
        return True
        
    def _initialize_logs(self, include_time=True):
        """
        Initializes empty log

        Args:
            include_time (bool): option to include timestamps or not

        Returns:
            (dict, list or None): Returns tuple of dictionary (value log) and list (time log) or None if include_time == False
        """
        log = {tuple(mdpt): [] for mdpt in self.mdpts}
        time_log = None
        if include_time:
            time_log = []
        temp_log = []
        return log, time_log, temp_log
        
    def _initialize_mdpts(self, img: ImageInfo):
        """
        Initializes midpoints at initialization from frame

        Args:
            img (ImageInfo): ImageInfo object

        Returns:
            (list???): list of midpoints detected in img
        """
        return img.get_midpoints()
    
    def _initialize_px2len_rate(self, img:ImageInfo):
        """
        Initializes pixel to length ratio

        Args:
            img (ImageInfo): ImageInfo object

        Returns:
            float: the conversion value to multiply by to go from pixels -> length
        """
        return img.get_px2len_rate()
        box = img.get_boxes()[self.ref_index]
        ax = img.get_axes()[self.ref_index]
        mdpt = img.get_midpoints()[self.ref_index]
        _, height, _, _ = func_utils.width_height(box, ax, mdpt)
        len_per_pixel = REFERENCE_OBJECT_LENGTH / height
        return len_per_pixel
        
    def measure_objects(self, img:ImageInfo):
        """
        Measures objects in img

        Args:
            img (ImageInfo): ImageInfo object

        Returns:
            (list of lines, list of float, list of mdpts): tuple of list of lines, list of distances, and list of midpoints
        """
        
        if len(img.get_axes()) != self.n_total_obj:
            warnings.warn(f'Detected {len(img.get_axes())} objects but expected {self.n_total_obj}')
        # else:
        lines, min_dists_px, mdpts = img.measure_objects()
        return lines, min_dists_px, mdpts
    
    def display_measure(self, img:ImageInfo, proc_img, in_unit=True, display_line = False):
        """
        Returns image with measures written on

        Args:
            img (ImageInfo): ImageInfo to pull information from 
            proc_img (matrix): image to draw values on
            in_unit (bool, optional): option to get values in pixels or length True->in units, False -> in pixels. Defaults to True.
            display_line (bool, optional): option to display line we are measuring on image or not. True-> display line, False-> don't display line Defaults to False.

        Returns:
            _type_: _description_
        """
        lines, min_dists, _ = self.measure_objects(img)
        if in_unit:
            min_dists = min_dists * self.px2len_rate
        if display_line:
            proc_img = img.get_line_draw_img(lines, proc_img)
        proc_img = img.get_display_vals_img(proc_img, min_dists)
        return proc_img
    
    def log_values(self, img:ImageInfo):
        """
        Adds values to log. Does not log if number of detected images does not equal original number of objects at initializaiton from frame

        Args:
            img (ImageInfo): ImageInfo object
        """
        _, min_dists, mdpts = self.measure_objects(img)
        if len(mdpts) != len(self.get_mdpts()):                 # ignore frames where there is inconsistent number of objects, this is typically due to artifcating so any measurements from these frames are typically not useable
            return
        set_mdpts = np.array(self.get_mdpts())
        for dist, pt in zip(min_dists, mdpts):
            closest_mdpt = self._closest_point_to_mdpts(pt, set_mdpts)
            self.log_by_mdpt[tuple(closest_mdpt)].append(dist)
        if self.time_log is not None:
            self.time_log.append(time.time() - self.start_time)
        # temp = self.read_thermometer()
        # if temp is not None:
        #     self.temp_log.append(temp)
        
    def _closest_point_to_mdpts(self, pt, mdpts):
        """
        Returns closest point in mdpts to pt

        Args:
            pt (coordinate): a point (x,y)
            mdpts (list of coordinates): list of points [(x,y), (x,y), ...]

        Returns:
            coordinate: the point in mdpts closest to pt
        """
        idx = np.argmin(np.linalg.norm(mdpts-pt, axis=1))
        return mdpts[idx]
    
    def log_to_dataframe(self, filename="log.csv", save_file = True):
        """
        Converts log to dataframe and saves in OUTPUT_DIR/filename if save_file==True

        Args:
            filename (str, optional): file to name csv file. Defaults to "log.csv".
            save_file (bool, optional): option to save as csv file or not
        """
        log_df = pd.DataFrame()
        if self.time_log is not None:
            log_df['time'] = self.time_log
        sorted_keys = sorted(list(self.log_by_mdpt.keys()), key=lambda x: x[0])
        for i in range(self.n_total_obj):
            if i == REFERENCE_INDEX:
                continue
            log_df[STENT_NAMES[i]] = self.log_by_mdpt[sorted_keys[i]] * np.array(self.px2len_rate)
        if save_file:
            log_df.to_csv(OUT_DIR / filename)
        return log_df
        
        
    def get_mdpts(self):
        """
        Returns originally initialized midpoints

        Returns:
            ???: the midpoints initialized from initialize_from_frame
        """
        return self.mdpts
    
    def read_thermometer(self):
        """
        Assuming that a thermometer is being read to a pc, this method returns the value being passed to 
        the pc. For logging purposes.

        Args:
            img (_type_): _description_
        """
        pass
    
    def run_experiment(self):
        cap = cv.VideoCapture(0)
        cv.namedWindow('vid1', cv.WINDOW_NORMAL)
        cv.namedWindow('vid2', cv.WINDOW_NORMAL)
        while cap.isOpened():
            
            rval, frame = cap.read()
            cv.imshow('vid1', frame)
            img = ImageInfo(frame)
            
            cont_morph_box = img.get_boxed_image(img.get_contoured_image(img.get_morph_image()), box_color=(0,0,255), box_thickness=3)
            cv.imshow('vid2', cont_morph_box)
            if cv.waitKey(1) == ord(' '):
                img = ImageInfo(frame) 
                self.initialize_from_frame(img, INCLUDE_TIME_LOG)
                break
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("broken or vid_end")
                break
            cv.imshow('vid1', frame)
            if cv.waitKey(1) == ord('q'):
                break
            
            img = ImageInfo(frame)
            cont_mor = img.get_contoured_image(img.get_morph_image())
            cont_mor = cv.cvtColor(cont_mor, cv.COLOR_GRAY2BGR)
            cont_mor = self.display_measure(img, cont_mor, display_line=True)
            #boxed = img.get_boxed_image(cont_mor, box_color=(0,0,255))
            cv.imshow('vid2', cont_mor)
            self.log_values(img)
        
        self.log_to_dataframe()
        
        cap.release()
        
        