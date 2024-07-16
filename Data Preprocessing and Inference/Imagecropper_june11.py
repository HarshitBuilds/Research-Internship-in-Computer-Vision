import cv2
import numpy as np
import os

# Change the directories as per need
###############################################################################
input_directory = r"F:\annotationapp\Data_Brazen\Batch 1 Sample 1\Top View Brazen Metal"
out =  r"F:\annotationapp\Data_Brazen\Batch 1 Sample 1\Top View Brazen Metal\cropped"
###############################################################################


# Output directory for cropped images
output_directory = os.path.join(out)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get a list of TIFF files in the input directory
tif_files = [file for file in os.listdir(input_directory) if file.endswith('.tif')]
print(len(tif_files))

for file_name in tif_files:
    # Load the image
    image_path = os.path.join(input_directory, file_name)
    image = cv2.imread(image_path)
    
# Paste the dimensions from Crop_Visualisation
###############################################################################    
    # Crop the image
    # cropped_image = image[626:2734, 1026:3134] #B2S1
    # cropped_image = image[1000:3024, 930:2954] #B2S2
    # cropped_image = image[938:2986, 2168:4216] #B1S2
    # cropped_image = image[638:2686, 1368:3416] #B1S3
    # cropped_image = image[276:2324, 788:2836]  #B1S5
    cropped_image = image[1350:3398, 4150:6198]  #B1S1

###############################################################################    
    # Output file path
    output_path = os.path.join(output_directory, f"{file_name}")

    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)

print("Cropping completed.")