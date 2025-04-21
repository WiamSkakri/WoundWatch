import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from model_resnet import get_resnet_model
import torch.nn as nn
import torch.optim as optim

# 1. Set the path to your dataset
data_dir = os.path.join(os.path.dirname(__file__), "data")

# 2. Define your transforms (ResNet expects 224x224, normalized)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet means
                         [0.229, 0.224, 0.225])  # ImageNet stds
])

# 3. Load dataset using ImageFolder
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 4. Split into train/val (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 5. Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 6. Print class names to check
print("Classes:", dataset.classes)
print("Number of training samples:", len(train_dataset))
print("Number of validation samples:", len(val_dataset))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_resnet_model(num_classes=2)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")
# Save the trained model
torch.save(model.state_dict(), "resnet_woundwatch.pth")
print("Model saved as resnet_woundwatch.pth")

