
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F

from model import create_model
from train import val_transform, MODEL_PATH

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

class_names = checkpoint["class_names"]

model = create_model(
    num_classes=len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


def predict(image):

    if image is None:
        return {}

    # PIL görüntüsü -> Tensor
    image = image.convert("RGB")
    image_tensor = val_transform(image)

    # Batch boyutu ekle: [1, 3, 224, 224]
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Sadece tahmin yap
    with torch.inference_mode():

        logits = model(image_tensor)

        probabilities = F.softmax(
            logits, dim=1
        )

    # 10 sınıfın skorlarını Python listesine çevir
    probabilities = probabilities[0].cpu().tolist()

    return dict(zip(class_names, probabilities))


# 5. Web arayüzü
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="Hayvan fotoğrafı yükle"
    ),
    outputs=gr.Label(
        num_top_classes=3,
        label="Tahmin sonuçları"
    ),
    title="AnimalClassifier",
    description=(
        "ResNet18 tabanlı yapay zekâ modeliyle "
        "10 farklı hayvan türünü sınıflandırın."
    )
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
