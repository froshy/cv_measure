from configuration import *
from classImageInfo import *
import measure_stent
import func_utils
import pandas as pd
import numpy as np
import warnings
import time

class MeasureExperiment():
    
    def __init__(self):
        self.num_stents = NUM_STENTS
        self.total_obj = N_TOTAL_OBJECTS
        self.ref_index = REFERENCE_INDEX
        self.px2len_rate = None
        self.mdpts = None                   # These midpoints correspond to each object, if we detect less objects in the next frame, 
                                            # the detected objects midpoints will correspond to the midpoint that is closest in the existing one.
                                            # if we detect more objects than we initialized, then the objects farthest away from the midpoints will be ignored
        self.log_by_mdpt = None             # initialized as None, but will be a dictionary where key is the midpoint (tuple) and values are a log of lengths associated with the object of the midpoint
        self.time_log = None
        self.start_time = None
    def initialize_from_frame(self, img:ImageInfo, include_time = True):
        assert len(img.get_axes()) == self.total_obj, f'Detected {len(img.get_axes())} objects but expected {self.total_obj}'
        self.px2len_rate = self._initialize_px2len_rate(img)
        self.mdpts = self._initialize_mdpts(img)
        self.log_by_mdpt, self.time_log = self._initialize_logs(include_time)
        self.start_time = time.time()
        
    def _initialize_logs(self, include_time):
        log = {tuple(mdpt): [] for mdpt in self.mdpts}
        time_log = None
        if include_time:
            time_log = []
        return log, time_log
        
    def _initialize_mdpts(self, img: ImageInfo):
        return img.get_midpoints()
    
    def _initialize_px2len_rate(self, img:ImageInfo):
        box = img.get_boxes()[self.ref_index]
        ax = img.get_axes()[self.ref_index]
        mdpt = img.get_midpoints()[self.ref_index]
        _, height, _, _ = func_utils.width_height(box, ax, mdpt)
        len_per_pixel = REFERENCE_OBJECT_LENGTH / height
        return len_per_pixel
        
    def measure_objects(self, img:ImageInfo):
        if len(img.get_axes()) != self.total_obj:
            warnings.warn(f'Detected {len(img.get_axes())} objects but expected {self.total_obj}')
        # else:
        #     self.mdpts = self._initialize_mdpts(img)
        lines, min_dists_px, mdpts = measure_stent.measure_stent(img, img.get_contoured_image(img.get_morph_image()))
        return lines, min_dists_px, mdpts
    
    def display_measure(self, img:ImageInfo, proc_img, in_unit=True, display_line = False):
        lines, min_dists, _ = self.measure_objects(img)
        if in_unit:
            min_dists = min_dists * self.px2len_rate
        if display_line:
            proc_img = img.get_line_draw_img(lines, proc_img)
        proc_img = img.get_display_vals_img(proc_img, min_dists)
        return proc_img
    
    def log_values(self, img:ImageInfo):
        _, min_dists, mdpts = self.measure_objects(img)
        if len(mdpts) != len(self.get_mdpts()):                 # ignore frames where there is inconsistent number of objects, this is typically due to artifcating so any measurements from these frames are typically not useable
            return
        set_mdpts = np.array(self.get_mdpts())
        for dist, pt in zip(min_dists, mdpts):
            closest_mdpt = self._closest_point_to_mdpts(pt, set_mdpts)
            self.log_by_mdpt[tuple(closest_mdpt)].append(dist)
        if self.time_log is not None:
            self.time_log.append(time.time() - self.start_time)
        #             dists = np.linalg.norm(set_mdpts[i] - mdpts[i], axis=1)
        #             closest_idx = np.argmin(dists)
        #             idx_arr[closest_idx] = i
        # for dist, mdpt in zip(min_dists, mdpts):
        #     cl_mdpt = self._closest_point_to_mdpts(mdpt, self.mdpts)
        #     self.log_by_mdpt[tuple(cl_mdpt)].append(dist)
        #     if self.log_time is not None:
        #         self.log_time.append(time.time() - self.start_time)
        
    def _closest_point_to_mdpts(self, pt, mdpts):
        idx = np.argmin(np.linalg.norm(mdpts-pt, axis=1))
        return mdpts[idx]
    
    def log_to_csv(self, filename="log.csv"):
        log_df = pd.DataFrame({'time': self.time_log})
        sorted_keys = sorted(list(self.log_by_mdpt.keys()), key=lambda x: x[0])
        for i in range(N_TOTAL_OBJECTS):
            if i == REFERENCE_INDEX:
                continue
            log_df[STENT_NAMES[i]] = self.log_by_mdpt[sorted_keys[i]] * np.array(self.px2len_rate)
        log_df.to_csv(OUT_DIR / filename)
        
        
    def get_mdpts(self):
        return self.mdpts