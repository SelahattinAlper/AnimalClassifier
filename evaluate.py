
from pathlib import Path
import time

import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from datasets import load_from_disk

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from model import create_model
from train import AnimalDataset, val_transform, DATA_PATH, MODEL_PATH


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Kullanılan cihaz:", device)

    # Kaydedilmiş en iyi modeli yükle
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

    # Test verilerini hazırla
    dataset = load_from_disk(str(DATA_PATH))

    test_data = AnimalDataset(
        dataset["test"],
        val_transform
    )

    test_loader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # Gerçek etiketler ve tahminler
    y_true = []
    y_pred = []

    with torch.inference_mode():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nTest Accuracy: {accuracy:.2%}")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d"
    )

    plt.title("AnimalClassifier - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        results_dir / "confusion_matrix.png",
        dpi=200
    )
    plt.close(fig)

    single_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    inference_times = []

    with torch.inference_mode():

        for i, (images, _) in enumerate(single_loader):

            images = images.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            if i >= 10:
                inference_times.append(elapsed)

            if len(inference_times) == 100:
                break

    avg_inference_ms = (
        sum(inference_times) / len(inference_times)
    ) * 1000

    print(
        f"\nOrtalama inference süresi: "
        f"{avg_inference_ms:.2f} ms/görüntü"
    )

    report_text = (
        f"Test Accuracy: {accuracy:.4f}\n"
        f"Ortalama model inference suresi: "
        f"{avg_inference_ms:.2f} ms/goruntu\n\n"
        f"{report}"
    )

    (results_dir / "evaluation_report.txt").write_text(
        report_text,
        encoding="utf-8"
    )

    print("\nTest tamamlandı!")

if __name__ == "__main__":
    main()
