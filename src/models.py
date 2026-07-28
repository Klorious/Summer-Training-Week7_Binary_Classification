import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights

def get_binary_model(model_name="resnet18", pretrained=True):
    if model_name == "vgg16":
        weights = VGG16_Weights.DEFAULT if pretrained else None
        model = vgg16(weights=weights)
        # VGG16 的最後一層在 classifier 的第 6 個 index
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, 2)
        
    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        # ResNet 系列的最後一層叫做 fc
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        
    else:
        raise ValueError(f"不支援的模型名稱: {model_name}")
        
    return model
