from pathlib import Path
import csv
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_from_disk
from model import create_model

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "animals10"
MODEL_PATH = ROOT / "models" / "best_model.pth"
HISTORY_PATH = ROOT / "results" / "training_history.csv"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class AnimalDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample["image"].convert("RGB")
        label = int(sample["label"])
        image = self.transform(image)
        return image,label

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        predictions = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size
    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy
def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

def main():

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Kullanılan cihaz:", device)

    dataset = load_from_disk(str(DATA_PATH))

    class_names = dataset["train"].features["label"].names

    print("Sınıflar:", class_names)

    train_data = AnimalDataset(
        dataset["train"], train_transform
    )

    val_data = AnimalDataset(
        dataset["validation"], val_transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    model = create_model(
        num_classes=len(class_names)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE
    )

    best_val_accuracy = -1.0
    history = []

    MODEL_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    HISTORY_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    start_time = time.perf_counter()

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion,
            optimizer, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        history.append([
            epoch + 1,
            train_loss,
            train_acc,
            val_loss,
            val_acc
        ])

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%}"
        )
        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%}"
        )

        if val_acc > best_val_accuracy:

            best_val_accuracy = val_acc

            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "best_val_accuracy": best_val_accuracy,
                "architecture": "resnet18"
            }, MODEL_PATH)

            print("Yeni en iyi model kaydedildi!")

    elapsed = time.perf_counter() - start_time

    with HISTORY_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy"
        ])
        writer.writerows(history)

    print("\nEğitim tamamlandı!")
    print(f"Toplam eğitim süresi: {elapsed:.1f} saniye")
    print(f"En iyi validation accuracy: {best_val_accuracy:.2%}")
    print("Model dosyası:", MODEL_PATH)
    print("Eğitim geçmişi:", HISTORY_PATH)


if __name__ == "__main__":
    main()
