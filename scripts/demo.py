from classImageInfo import *
from classExperiment import *
import cv2 as cv 
import os
import glob
def foo():
    img_dir = "./img/"
    
    for filename in glob.glob(img_dir + '*.jpg'):
        if os.path.basename(filename) in ['20m53.jpg', 'white_bg.jpg']:
            continue
        print(filename)
        img = cv.imread(filename)
        img = ImageInfo(img)
        measured = img.get_measured_image()
        cv.namedWindow(os.path.basename(filename), cv.WINDOW_NORMAL)
        cv.imshow(os.path.basename(filename), measured)
        
    cv.waitKey(0)
    
foo()
        