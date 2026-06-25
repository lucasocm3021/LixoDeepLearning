from pathlib import Path

import torch
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "reciclaai.pth"

BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.001
NUM_WORKERS = 4  # reduza para 0 no Windows se tiver problemas de multiprocessing


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # entradas sempre 224x224 -> deixa o cuDNN escolher o melhor algoritmo
        torch.backends.cudnn.benchmark = True
    return device


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, test_transform


def load_data(device: torch.device) -> tuple[DataLoader, DataLoader, list[str]]:
    train_transform, test_transform = build_transforms()

    train_dataset = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transform)
    test_dataset = datasets.ImageFolder(DATASET_DIR / "test", transform=test_transform)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )

    return train_loader, test_loader, train_dataset.classes


def build_model(class_count: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, class_count)

    return model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    average_loss = total_loss / len(loader)

    return average_loss, accuracy

def plot_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    classes: list[str],
    device: torch.device,
) -> None:
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            predictions = outputs.argmax(1).cpu().tolist()

            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    matrix = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes,
    )

    display.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png")
    plt.show()

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    average_loss = total_loss / len(loader)

    return average_loss, accuracy


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    device = get_device()
    train_loader, test_loader, classes = load_data(device)

    model = build_model(len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(device.type) if device.type == "cuda" else None

    print(f"Device: {device}")
    print(f"Classes: {classes}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler, 
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2%} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.2%}"
        )
    plot_confusion_matrix(model, test_loader, classes, device)
    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": classes,
        },
        MODEL_PATH,
    )

    print(f"Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
