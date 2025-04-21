import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_resnet_model(num_classes=2):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # ✅ Freeze all layers *except* the last block (layer4)
    for name, param in model.named_parameters():
        if not name.startswith("layer4"):
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

