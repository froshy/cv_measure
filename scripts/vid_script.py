import cv2 as cv 
import time
from classImageInfo import *
import func_utils
# import measure_stent
from classExperiment import *
def foo():
    filename = "./vid/big_stent_vid.mkv"
    test = MeasureExperiment()
    cap = cv.VideoCapture(filename) 
    cv.namedWindow('vid', cv.WINDOW_NORMAL)
    cv.namedWindow('vid2', cv.WINDOW_NORMAL)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[330:920, 300:1500, :] 
        if not ret:
            break
        cv.imshow('vid', frame)
        img = ImageInfo(frame)
        cont_morph_box = img.get_boxed_image(img.get_contoured_image(img.get_morph_image()), box_color=(0,0,255), box_thickness=3)
        cv.imshow('vid2', cont_morph_box)
        if cv.waitKey(1) == ord(' '):
            img = ImageInfo(frame) 
            test.initialize_from_frame(img)
            break
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[330:920, 300:1500, :] 
        if not ret:
            print("broken or vid_end")
            break
        cv.imshow('vid', frame)
        if cv.waitKey(1) == ord('q'):
            break
        
        img = ImageInfo(frame)
        cont_mor = img.get_contoured_image(img.get_morph_image())
        cont_mor = cv.cvtColor(cont_mor, cv.COLOR_GRAY2BGR)
        cont_mor = test.display_measure(img, cont_mor, display_line=True)
        #boxed = img.get_boxed_image(cont_mor, box_color=(0,0,255))
        cv.imshow('vid2', cont_mor)
        
    
    cap.release()
    cv.destroyAllWindows
    

foo()