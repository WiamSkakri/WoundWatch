import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load your trained model
from model_resnet import get_resnet_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet_model(num_classes=2)
model.load_state_dict(torch.load("resnet_woundwatch.pth", map_location=device))
model.eval()
model.to(device)

# Transform for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Hooks to store data
features = []
gradients = []

def save_features_hook(module, input, output):
    features.clear()
    features.append(output.clone())  # <- clone avoids view issues

def save_gradients_hook(module, grad_input, grad_output):
    gradients.clear()
    gradients.append(grad_output[0].clone())  # <- clone avoids view issues

# ✅ Use conv2 (not relu) to avoid in-place problems
target_layer = model.layer4[1].conv2
target_layer.register_forward_hook(save_features_hook)
target_layer.register_full_backward_hook(save_gradients_hook)

def generate_gradcam(image_path):
    print(f"🔍 Loading image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Forward pass
    print("➡️ Running forward pass...")
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    print("⬅️ Running backward pass...")
    output[0, pred_class].backward()
    print("✅ Gradients captured:", len(gradients) > 0)

    # Get stored values
    fmap = features[0]
    grads = gradients[0]

    pooled_grads = torch.mean(grads, dim=[0, 2, 3])

    weighted_fmap = fmap[0].clone()
    for i in range(weighted_fmap.shape[0]):
        weighted_fmap[i, :, :] *= pooled_grads[i]

    cam = torch.sum(weighted_fmap, dim=0)
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max()
    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))

    # Overlay heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    original = np.array(img.resize((224, 224))) / 255.0
    overlay = heatmap + original
    overlay = overlay / np.max(overlay)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(f"Grad-CAM: Class {pred_class}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "data/infected/infected_01.png"  # Update with your image path
    generate_gradcam(image_path)

