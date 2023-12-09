from google.colab import drive
drive.mount('/content/drive')

# Install necessary libraries
!pip install torch torchvision

# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import Resize
import os


# Both blurry and noisy
# Define U-Net architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name)

        # Convert grayscale images to RGB
        if image.mode == 'L':
            image = image.convert('RGB')

        # Remove alpha channel
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

# Set your dataset path
data_path = "/content/drive/MyDrive/images"

# Define data transformations
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust the size as needed
    transforms.ToTensor(),
])

# Create dataset instances
blurry_dataset = CustomDataset(os.path.join(data_path, "blurry"), transform=data_transform)
noisy_dataset = CustomDataset(os.path.join(data_path, "noisy"), transform=data_transform)

# Create DataLoader instances
blurry_loader = DataLoader(blurry_dataset, batch_size=16, shuffle=True)
noisy_loader = DataLoader(noisy_dataset, batch_size=16, shuffle=True)

# Initialize your model
model = UNet()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    blurry_iter = iter(blurry_loader)
    noisy_iter = iter(noisy_loader)

    total_loss = 0.0

    for _ in range(min(len(blurry_loader), len(noisy_loader))):
        # Ensure both batches have the same number of channels and size
        blurry_batch = next(blurry_iter)
        noisy_batch = next(noisy_iter)

        # Adjust the batch size based on the smaller of the two batches
        min_batch_size = min(blurry_batch.size(0), noisy_batch.size(0))
        blurry_batch = blurry_batch[:min_batch_size, :, :, :]
        noisy_batch = noisy_batch[:min_batch_size, :, :, :]

        # Resize the noisy_batch tensor to match the size of blurry_batch
        resize_transform = Resize(blurry_batch.shape[-2:])
        noisy_batch = torch.stack([resize_transform(img) for img in noisy_batch])

        # Forward pass
        output = model(noisy_batch)

        # Calculate loss
        loss = criterion(output, blurry_batch)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / min(len(blurry_loader), len(noisy_loader))
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")

print("Training finished.")




# only blurry

# Define U-Net architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name)

        # Convert grayscale images to RGB
        if image.mode == 'L':
            image = image.convert('RGB')

        # Remove alpha channel
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

# Set your dataset path
data_path = "/content/drive/MyDrive/GOPRO_Large/train/GOPR0884_11_00"

# Define data transformations
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust the size as needed
    transforms.ToTensor(),
])

# Create dataset instance for blurry images
blurry_dataset = CustomDataset(os.path.join(data_path, "blur"), transform=data_transform)

# Create DataLoader instance for blurry images
blurry_loader = DataLoader(blurry_dataset, batch_size=16, shuffle=True)

# Initialize your model
model = UNet()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop for blurry images
num_epochs = 10

for epoch in range(num_epochs):
    blurry_iter = iter(blurry_loader)

    total_loss = 0.0

    for _ in range(len(blurry_loader)):
        blurry_batch = next(blurry_iter)

        # Forward pass
        output = model(blurry_batch)

        # Calculate loss
        loss = criterion(output, blurry_batch)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(blurry_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")

print("Training finished.")




# Import necessary libraries for testing
import matplotlib.pyplot as plt
import numpy as np

# Function to display images
def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()

# Set your test dataset path
test_data_path = "/content/drive/MyDrive/images/blurry"

# Create a DataLoader for the test dataset
test_dataset = CustomDataset(test_data_path, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Set the model to evaluation mode
model.eval()

# Test loop
with torch.no_grad():
    for test_batch in test_loader:
        # Forward pass
        output = model(test_batch)

        # Display original and reconstructed images
        print("Original Image:")
        imshow(test_batch.squeeze())

        print("Reconstructed Image:")
        imshow(output.squeeze())
