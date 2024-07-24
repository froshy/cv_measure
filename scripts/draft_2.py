from classImageInfo import *
import measure_stent
import cv2 as cv
import time
import os
import glob
import ms

def foo():
    file_path = "./img/19m58.jpg"
    img = ImageInfo(file_path)
    cv.imshow('foo', img.get_scan_box_image())
    cv.waitKey(0)
    
foo()
    