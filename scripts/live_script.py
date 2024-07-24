import cv2 as cv 
import time
from classImageInfo import *
import func_utils
import measure_stent
def foo():
    cap = cv.VideoCapture(0)
    rval, frame = cap.read()
    while rval:
        cv.imshow('vid', frame)
        rval, frame = cap.read()
        key = cv.waitKey(1)
        if key == 27:               #  press escape to leave
            break
        
        img = ImageInfo(frame)
        cont_mor = img.get_contoured_image(img.get_morph_image())
        lines, dists = measure_stent.measure_stent(img, cont_mor)
        cont_mor = cv.cvtColor(cont_mor, cv.COLOR_GRAY2BGR)
        for st, end in lines:
            cv.line(cont_mor, np.round(st).astype(int), np.round(end).astype(int), (0,0,255), 1)
        cv.imshow('vid2', cont_mor)
        
        
    
    cap.release()
foo()