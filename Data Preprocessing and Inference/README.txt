Correct Order to Use Python Scripts:

Crop_visualisation_june27.py ------------> To visualise the crop manually [(m x n) to (2048 x 2048)]
Imagecropper_june11.py ------------------> To crop images in an entire directory
Imagesplitter [Twice].py ----------------> To split the images into their respective quadrants [we do this twice to process the images into dimensions 512x512]
Empty_Image_deletor_june21.py -----------> To remove empty images which provide no use to quantification
SavingPredictions_Visualization.py ----> To save and visualize predicted masks using input images in sample
quantification_voids.py -----------------> To quantify the voids present in the sample


To use the scripts ,we have marked the lines of code that needs to be changed as per sample between lines of hashtags.