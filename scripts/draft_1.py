from classImageInfo import *
import cv2 as cv
import time
import os
import glob
#import ms

def foo():
    file_path = "./img/19m58.jpg"
    img = ImageInfo(file_path)
    # contoured = img.get_contoured_image(img.get_morph_image())
    # #contoured = cv.cvtColor(contoured, cv.COLOR_GRAY2BGR)
    # boxes = img.get_scan_box_image(contoured)
    # lines, min_dist = ms.measure_stent(img, contoured)
    # for st, end, in lines:
    #     cv.line(contoured, np.round(st).astype(int), np.round(end).astype(int), (0,0,255), 1)
    #     cv.line(boxes, np.round(st).astype(int), np.round(end).astype(int), (0,0,255), 1)
    # cv.imshow('cont', contoured)
    # cv.imshow('box', boxes)
    # cv.waitKey(0)
foo()