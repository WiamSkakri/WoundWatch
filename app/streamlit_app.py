import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "classification")))
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model_resnet import get_resnet_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "segmentation")))
from unet import UNet 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load segmentation model
seg_model = UNet()
seg_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "segmentation", "unet_wound_segmentation.pth"))
seg_model.load_state_dict(torch.load(seg_model_path, map_location=device))
seg_model.eval()
seg_model.to(device)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet_model(num_classes=2)
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "classification", "resnet_woundwatch.pth"))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

# Labels
labels = ['healing', 'infected']

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Hooks
features, gradients = [], []

def forward_hook(module, input, output):
    features.clear()
    features.append(output.clone())

def backward_hook(module, grad_input, grad_output):
    gradients.clear()
    gradients.append(grad_output[0].clone())

target_layer = model.layer4[1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# Grad-CAM
def get_gradcam(img_tensor, class_idx):
    model.zero_grad()
    output = model(img_tensor)
    output[0, class_idx].backward()

    fmap = features[0]
    grads = gradients[0]
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])

    for i in range(fmap.shape[1]):
        fmap[0, i, :, :] *= pooled_grads[i]

    cam = torch.sum(fmap, dim=1).squeeze()
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max()
    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return heatmap

# Streamlit UI
st.set_page_config(page_title="WoundWatch", layout="wide")
st.title("🩹 WoundWatch: AI-Powered Wound Classifier")

uploaded_file = st.file_uploader("Upload a wound image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img_tensor = transform(image).unsqueeze(0).to(device)
    output = model(img_tensor)
    pred_class = torch.argmax(output, dim=1).item()
    st.markdown(f"### 🔍 Prediction: **{labels[pred_class]}**")

    # Grad-CAM
    heatmap = get_gradcam(img_tensor, pred_class)
    overlay = np.array(image.resize((224, 224))) / 255.0 + heatmap / 255.0
    overlay = overlay / overlay.max()

    col1, col2 = st.columns(2)
    with col1:
        st.image(image.resize((224, 224)), caption="Original Image")
    with col2:
        st.image(overlay, caption="Grad-CAM Heatmap")
    st.markdown("### 🧠 Segmentation Output")

    # Segmentation transform
    seg_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    seg_input = seg_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        seg_output = seg_model(seg_input)
        seg_mask = seg_output.squeeze().cpu().numpy()
        seg_mask = (seg_mask > 0.5).astype(np.uint8)

    # Overlay mask
    resized_img = np.array(image.resize((256, 256)))
    mask_overlay = resized_img * seg_mask[:, :, None]

    col3, col4 = st.columns(2)
    with col3:
        st.image(seg_mask * 255, caption="Predicted Segmentation Mask", use_container_width=True)
    with col4:
        st.image(mask_overlay.astype(np.uint8), caption="Overlay on Image", use_container_width=True)


