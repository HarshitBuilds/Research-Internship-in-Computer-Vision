import cv2
import os

###############################################################################
### Use this code to identify the correct cropping dimensions
### make sure to either crop the sample across 2048 x 2048 or 512*(multiplier) x 512*(multiplier)
### The model takes input (512 x 512) images
###############################################################################

# change image location and indices to identify a good crop
###############################################################################
image_path = r"F:/annotationapp/Data_Brazen/Batch 1 Sample 1/Top View Brazen Metal/10030.tif"
image = cv2.imread(image_path)
cropped_image = image[1350:3398, 4150:6198] ## image[along y axis , along x axis]
###############################################################################
cv2.imshow("Cropped",cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
