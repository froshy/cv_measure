from pathlib import Path

# Set up directories
IMG_DIR = Path("./img/")                         # directory to image (UNUSED)
OUT_DIR = Path("./output/")                      # directory to output (UNUSED)

# Set up image processing for edge detection
MORPHOLOGICAL_ITERS = 10                         # Number of iterations of dilate and erode to morph image
CANNY_TH1 = 20                                   # Define Canny thresholds
CANNY_TH2 = 45                                   # Define Canny thresholds
CANNY_SIGMA = 0.33                               # For automatically defining Canny thresholds
CONTOUR_AREA_THRESHOLD = 3000                    # Contours with area < CONTOUR_THRESHOLD are ignored when adding boxes/processing

# Set up testing environment variables  
NUM_STENTS = 4                                   # number of Stents in experiment
STENT_NAMES = ['ref', 'a','b','c','d','e','f','g','h'] # Names of the stents, use placeholder for reference index
REFERENCE_OBJECT_LENGTH = 30                     # the dimension of the reference object; reference object should be rectangular and the side of longest length goes here
UNITS = "mm"                                     # Units of length, not used for calculations, but used for displaying
REFERENCE_INDEX = 0                              # The position of the reference object, going left to right. Use 0 for left-most object and -1 for right-most object
INCLUDE_TIME_LOG = True                          # Option to include a time log or not
# Some parameters of the program
# do not change unless you know how these work
BLACK_PIXEL_THRESHOLD = 10                       # max value to be considered a black pixel in single channel image (range 0-255)
WHITE_PIXEL_THRESHOLD = 240                      # min value to be considered a white pixel in single channel image (range 0-255)
SCAN_WIDTH_TOL = 1.2                             # > 1 for wider than width, < 1 for thinner than width, the tolerance when scanning for width, will search SCAN_WIDTH_TOL times wider to either side of the midpoint (perpendicular to orientation)
SCAN_HEIGHT_PROP = 0.1                           # the proportion of height to use when scanning for width, i.e. will scan SCAN_HEIGHT_PROP*height above midpoint and SCAN_HEIGHT_PROP*height below midpoint (parallel to orientation)
SCAN_FREQ = 3                                    # The number of scans to perform per side in scan box, one scan scans pixels going to midpoint perpendicular to axis, 2 scans scan the area into 'thirds'
COLOR_STENT = (0,0,255)                          # color for the stent so we can identify what is 'stent' and what isn't in the photo
N_TOTAL_OBJECTS = NUM_STENTS + 1