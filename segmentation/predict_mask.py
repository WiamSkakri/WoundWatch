import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from unet import UNet

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()
model.load_state_dict(torch.load("unet_wound_segmentation.pth", map_location=device))
model.eval()
model.to(device)

# Transforms
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load test image
test_img_path = "../data/test/Images/fusc_0275.png"
image = Image.open(test_img_path).convert("RGB")
input_tensor = transform_img(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    mask_pred = output.squeeze().cpu().numpy()
    mask_pred = (mask_pred > 0.5).astype(np.uint8)

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(mask_pred, cmap="gray")
plt.title("Predicted Mask")

plt.subplot(1, 3, 3)
overlay = np.array(image.resize((256, 256))) * mask_pred[:, :, None]
plt.imshow(overlay.astype(np.uint8))
plt.title("Overlay")
plt.tight_layout()
plt.show()

