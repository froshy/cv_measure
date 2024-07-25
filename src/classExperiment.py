from configuration import *
from classImageInfo import *
import measure_stent
import func_utils

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
    
    def initialize_from_frame(self, img:ImageInfo):
        assert len(img.get_axes()) == self.total_obj, f'Detected {len(img.get_axes())} objects but expected {self.total_obj}'
        self.px2len_rate = self._initialize_px2len_rate(img)
        self.mdpts = self._initialize_mdpts(img)
        self.log_by_mdpt = self._initialize_log_by_mdpt()
        #self.last_frame = img
        
    def _initialize_log_by_mdpt(self):
        log = {tuple(mdpt): [] for mdpt in self.mdpts}
        return log
        
        
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
        assert len(img.get_axes()) == self.total_obj, f'Detected {len(img.get_axes())} objects but expected {self.total_obj}'
        lines, min_dists_px = measure_stent.measure_stent(img, img.get_contoured_image(img.get_morph_image()))
        return lines, min_dists_px
    
    def display_measure(self, img:ImageInfo, proc_img, in_unit=True, display_line = False):
        lines, min_dists = self.measure_objects(img)
        if in_unit:
            min_dists = min_dists * self.px2len_rate
        if display_line:
            proc_img = img.get_line_draw_img(lines, proc_img)
        proc_img = img.get_display_vals_img(proc_img, min_dists)
        return proc_img