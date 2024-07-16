import os
import cv2

###############################################################################
### To convert the 2048x2048 images to 512x512, 
### use this script TWICE as this code is used to cut up a image into its 4 quadrants
###############################################################################

def split_images_in_folder_and_save(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):  # Process only TIFF files
            # Read the image
            image_path = os.path.join(input_folder, filename)
            original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            # Get the dimensions of the original image
            height, width = original_image.shape[:2]

            # Calculate the dimensions of each quadrant
            quadrant_width = width // 2
            quadrant_height = height // 2

            # Extract and save each quadrant
            for i in range(2):
                for j in range(2):
                    quadrant_name = f"{os.path.splitext(filename)[0]}_{i}{j}.tif"
                    quadrant_path = os.path.join(output_folder, quadrant_name)

                    # Extract the quadrant
                    left = j * quadrant_width
                    top = i * quadrant_height
                    right = (j + 1) * quadrant_width
                    bottom = (i + 1) * quadrant_height
                    quadrant = original_image[top:bottom, left:right]

                    # Save the quadrant
                    cv2.imwrite(quadrant_path, quadrant)


# Change directories as per use  
###############################################################################
input_path = r"F:\annotationapp\Data_Brazen\Batch 1 Sample 1\Top View Brazen Metal\cropped_1024"  # Replace this with the path to your image
output_dir = r"F:\annotationapp\Data_Brazen\Batch 1 Sample 1\Top View Brazen Metal\cropped_512"    # Name of the directory to save quadrants
###############################################################################
split_images_in_folder_and_save(input_path, output_dir)



