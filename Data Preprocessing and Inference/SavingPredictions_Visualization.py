import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
import numpy as np
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from PIL import Image

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

###############################################################################
### Used to predict the mask for untrained input
###############################################################################


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("L")  # Convert image to grayscale

        if self.transform:
            image = self.transform(image)

        return image

# Define transformations
transform = tt.Compose([
    tt.ToTensor()
])

# Input directory upon which predictions are to be made
###############################################################################
image_dir = r'F:\annotationapp\Data_Brazen\Batch 1 Sample 1\Top View Brazen Metal\cropped_512_without_blanks'
###############################################################################

dataset = CustomDataset(image_dir, transform)


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        def CBR(in_channels, out_channels):
          return nn.Sequential(
              nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(),
              nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU()
          )

        #Downsampling the data
        self.encoder1 = CBR(1, 64) #2048x2048x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #1024x1024x64
        self.encoder2 = CBR(64, 128) #1054x1054x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #512x512x128
        self.encoder3 = CBR(128, 256) #512x512x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) #256x256x512
        self.encoder4 = CBR(256, 512) #256x256x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) #128x128x512

        #Bottleneck containing 1024 channels.
        self.bottleneck = CBR(512, 1024) #128x128x1024

        #Upsampling the data
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) #256x256x512
        self.decoder4 = CBR(1024, 512) #256x256x512
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = CBR(128, 64)

        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)

    #Forward pass of the model
    def forward(self,x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv_last(dec1))

model = Unet()

# Change model and device on which the model runs here 
###############################################################################    
model.load_state_dict(torch.load("F:/annotationapp/models/UNET_512_DICE_largevoids.pth", map_location=torch.device('cpu')))
###############################################################################


device = torch.device("cpu")


from PIL import Image
from skimage.transform import resize  # Assuming you are using skimage for resizing

def saving_visualization_pred_images(image, predicted_mask, idx, output_dir, resize_dim=(512,512)):
    # Convert PyTorch tensors to NumPy arrays
    image_np = image.squeeze().numpy() * 255  # Assuming image is already in grayscale and in [0, 1] range
    predicted_mask_np = predicted_mask.numpy() * 255  # Convert predicted_mask to numpy array

    # Resize the image and masks
    resized_image_np = resize(image_np, resize_dim)
    resized_predicted_mask_np = resize(predicted_mask_np, resize_dim)

    # Convert to uint8
    resized_image_np = resized_image_np.astype(np.uint8)
    resized_predicted_mask_np = resized_predicted_mask_np.astype(np.uint8)

    # Create output directories if they don't exist
    os.makedirs(os.path.join(output_dir, "Images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Predicted_Image"), exist_ok = True)
    
    # Save images using PIL
    img_path = os.path.join(output_dir, "Images", f"Image_{idx}.tif")
    out_path = os.path.join(output_dir, "Predicted_Image", f"Predicted_mask_{idx}.tif")

    Image.fromarray(resized_image_np).save(img_path)
    Image.fromarray(resized_predicted_mask_np).save(out_path)



    # Optionally save ground truth mask
    # ground_truth_path = os.path.join(output_dir, "Ground_Truth_Mask", f"Ground_truth_mask_{idx}.png")
    # Image.fromarray(resized_mask_np).save(ground_truth_path)

# Example usage:
# Assuming 'image', 'mask', 'predicted_mask' are your tensors from dataset[idx]
# image_feed = image.unsqueeze(0).to(device)  # Assuming you add batch dimension and move to device
# prediction = model(image_feed)  # Assuming model outputs the predicted mask
# predicted_mask = prediction.squeeze().cpu().detach()


# Call the visualization function
# saving_visualization_pred_images(image, mask, predicted_mask, idx, output_dir)

# Output directory where the mask is to be stored
###############################################################################
output_dir = r"F:\annotationapp\Data_Brazen\Batch 1 Sample 1\Top View Brazen Metal\Predicted_masks"
###############################################################################

# Visualize selected images with their ground truth masks and predicted masks using OpenCV
for idx in range(len(dataset)):
    image = dataset[idx]
    image_feed = image.unsqueeze(0).to(device) # Add batch dimension and move to device   

    # Get prediction
    with torch.no_grad():
        prediction = model(image_feed)  # Assuming model outputs the predicted mask
        predicted_mask = prediction.squeeze().cpu().detach()  # Move to CPU, detach, and convert to numpy array
        
    saving_visualization_pred_images(image,predicted_mask,idx,output_dir)

