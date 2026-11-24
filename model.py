import torch
import torch.nn as nn
from torchvision.models import resnet18,ResNet18_Weights

def create_model(num_classes=10):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    return model
if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = create_model(num_classes=10)
    model = model.to(device)
    model.eval()

    print("Kullanılan cihaz:", device)
    print("Son katman:", model.fc)

    dummy_images = torch.randn(2, 3, 224, 224).to(device)

    with torch.inference_mode():
        outputs = model(dummy_images)

    print("Çıkış boyutu:", outputs.shape)