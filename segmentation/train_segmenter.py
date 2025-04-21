import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import WoundSegmentationDataset
from unet import UNet
from tqdm import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = WoundSegmentationDataset(
    image_dir="../data/train/Images",
    mask_dir="../data/train/Masks",
    transform_img=transform_img,
    transform_mask=transform_mask
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Model
model = UNet()
model.to(device)

# Loss & optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
print("🚀 Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "unet_wound_segmentation.pth")
print("✅ Model saved as unet_wound_segmentation.pth")

