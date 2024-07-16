import cv2
import numpy as np
import os 
import pandas as pd

###############################################################################
### using contours to identify the total number of voids
### identifing the total number of void pixels and sample pixels
###############################################################################

# Input prediction mask directory
###############################################################################
image_dir = r"F:\annotationapp\Data_Brazen\Batch 1 Sample 1\Top View Brazen Metal\Predicted_masks\Predicted_Image" # predicted mask directory
input_dir = r"F:\annotationapp\Data_Brazen\Batch 1 Sample 1\Top View Brazen Metal\Predicted_masks\Images" # input image directory
###############################################################################

# below code is made for complete samples [images are not being deleted] 
files = os.listdir(image_dir)

count = []
pixel = []
for file in files:
    file_path = os.path.join(image_dir, file)
    image = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in ['.tiff','.tif']):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = len(contours)
        count.append(c)
        _, binary_img = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        non_black_pixel_count = np.count_nonzero(binary_img)
        pixel.append(non_black_pixel_count)

total = []
files1 = os.listdir(input_dir)
for file in files1:
    file_path = os.path.join(input_dir,file)
    image = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in ['.tiff','.tif']):
        _, binary_img = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        sample_count = np.count_nonzero(binary_img)
        total.append(sample_count)        
        
########################### for complete samples ##############################

# total_images = len(count)//16 
# voids = []
# for i in range(total_images):
#     j1 = i*16
#     j2 = (i+1)*16
#     voids.append(np.sum(count[j1:j2]))


# df = pd.DataFrame({'Image': [f"image-{i+1}" for i in range(len(voids))],
#                    'Void Count': voids})

################## for samples which empty images are deleted #################

df = pd.DataFrame({'Image': [f"image-{i+1}" for i in range(len(count))],
                   'Void Count': count,
                   'Void Pixel Count': pixel,
                   'Total Sample Pixel Count' : total})


# Excel Directory where data is to be stored
###############################################################################
excel_path = r'C:\Users\Eyepod\Desktop\Ramakrishna\void_counts_B1S1.xlsx'
###############################################################################
 

df.to_excel(excel_path, index=False)

print(f"Data exported to {excel_path}")
        
        
    