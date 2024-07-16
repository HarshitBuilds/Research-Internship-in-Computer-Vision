import os
import cv2
import numpy as np

###############################################################################
### To reduce computational expense ,  we delete input images with nothing in them 
###############################################################################

# Function to delete files based on condition
def delete_files_based_on_condition(directory_path, condition):
    # List all files in the directory
    files = os.listdir(directory_path)
    
    for file in files:
        file_path = os.path.join(directory_path, file)
        
        # Check if the condition is satisfied
        if condition(file_path):
            try:
                # Attempt to remove the file
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")


def condition_file_size(file_path):
    image = cv2.imread(file_path)
    if len(np.unique(image)) == 1:
        return True
    else:
        return False

# Change the directory of images as per sample
###############################################################################
if __name__ == "__main__":
    directory_path = r"F:\annotationapp\Data_Brazen\Batch 1 Sample 1\Top View Brazen Metal\cropped_512_without_blanks"
    delete_files_based_on_condition(directory_path, condition_file_size)
###############################################################################