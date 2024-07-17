# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
import torch
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import cv2
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset,ConcatDataset

#loading the data

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("L")  # Convert image to grayscale
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Define transformations
transform = tt.Compose([
    tt.ToTensor()
])

# Create dataset and dataloader
image_dir = r'/kaggle/input/data-512/512x512/Training'
mask_dir = r'/kaggle/input/data-512/512x512/Mask'
dataset = CustomDataset(image_dir, mask_dir, transform)

# Define the split ratio
train_size = int(0.8 * len(dataset)) # 90% for training
aug_size = int(0 * len(dataset))  
val_size = len(dataset) - train_size - aug_size  # 10% for validation

seed = 42
torch.manual_seed(seed)
train_dataset,augmentation_dataset, val_dataset = random_split(dataset, [train_size,aug_size ,val_size])

aug_dir_training = r'/kaggle/input/largevoids/big_voids_samples/training'
aug_dir_mask = r'/kaggle/input/largevoids/big_voids_samples/mask'
aug_dataset = CustomDataset(aug_dir_training, aug_dir_mask, transform)

len(aug_dataset)

train_transform = tt.Compose([tt.RandomVerticalFlip(),
                         tt.RandomHorizontalFlip(),
                         #tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)),
                         tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
                        ])

augmented_samples = []

# Apply transformations to each sample in train_ds
for image, mask in aug_dataset:
    augmented_samples_for_image = []
    for _ in range(9):  # Define the number of augmented samples per original sample
        seed = torch.seed()  # Seed for reproducibility
        torch.manual_seed(seed)
        augmented_image = train_transform(image)

        torch.manual_seed(seed)
        augmented_mask = train_transform(mask)

        augmented_samples_for_image.append((augmented_image, augmented_mask))

    augmented_samples.extend(augmented_samples_for_image)
    # Convert augmented samples to tensors
augmented_images, augmented_masks = zip(*augmented_samples)
augmented_images = torch.stack(augmented_images)  # Convert list of tensors to a single tensor
augmented_masks = torch.stack(augmented_masks)  # Convert list of labels to a tensor
    # Create a TensorDataset
augmented_train_ds = TensorDataset(augmented_images, augmented_masks)

train_dataset = ConcatDataset([train_dataset, augmented_train_ds])
len(train_dataset)

batch_size = 4
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

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
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, target):
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice
    
model = Unet()

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

def train_model(model, train_loader, val_loader, epochs=10, accumulation_steps=6):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = DiceLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stopping = EarlyStopping(patience=5, delta=0.01)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Initialize accumulation variables
        optimizer.zero_grad()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for i, (image, mask) in progress_bar:
            image, mask = image.to(device), mask.to(device)
            output = model(image)
            loss = criterion(output, mask)
            loss.backward()

            if (i + 1) % accumulation_steps == 0:  # gradient accumulation
                optimizer.step()  # Update weights
                optimizer.zero_grad()  # Clear the gradients for the next accumulation

            train_loss += loss.item()

        # If there are remaining gradients to be updated
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, mask in val_loader:
                image, mask = image.to(device), mask.to(device)
                output = model(image)
                loss = criterion(output, mask)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    early_stopping.load_best_model(model)

train_model(model, train_loader,val_loader,epochs = 15)

torch.save(model.state_dict(), 'UNET_512_DICE_largevoids.pth')