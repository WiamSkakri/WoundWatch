import torch
from unet import UNet

# Try to load the model
try:
    model = UNet()
    model.load_state_dict(torch.load("unet_wound_segmentation.pth", map_location="cpu"))
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:")
    print(e)

