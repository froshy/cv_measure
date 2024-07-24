from classImageInfo import *
import measure_stent
import cv2 as cv
import time
import os
import glob

if __name__ == "__main__":
    
    st = time.time()
    for filename in glob.glob("img/*"):
        
        img = ImageInfo(filename)
        orig = img.get_image()
        lines, _ = measure_stent.measure_stent(img, img.get_contoured_image(img.get_morph_image()))
        for st, end in lines:
            cv.line(orig, np.round(st).astype(int), np.round(end).astype(int), (0,0,255), 1)
        cv.imshow(os.path.basename(filename), orig)
        
        # print(img.get_image().shape)
        # print(img.get_image().shape)
        # contoured_morph = cv.cvtColor(img.get_contoured_image(image=img.get_morph_image()), cv.COLOR_GRAY2BGR)
        # contoured_morph = img.get_contoured_image(image=img.get_morph_image())
        # boxed = img.get_scan_box_image(contoured_morph)
        # lines = measure_stent.measure_stent(img, contoured_morph)
        # print(time.time() - st)
    cv.waitKey(0)
        
    