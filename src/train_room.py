import os
import time
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from dataset import list_images, RoomsDataset
from model import RoomClassifier


CLASSES = ["Bathroom", "Bedroom", "Dinning", "Kitchen", "Livingroom"]
DATA_DIR = Path("data/raw/House_Room_Dataset")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 8
LR = 3e-4
IMG_SIZE = 224


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    samples = list_images(DATA_DIR, CLASSES)
    paths = [p for p, y in samples]
    labels = [y for p, y in samples]

    train_idx, temp_idx = train_test_split(
        np.arange(len(samples)), test_size=0.30, random_state=SEED, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=np.array(labels)[temp_idx]
    )

    def subset(idxs):
        return [samples[i] for i in idxs]

    train_samples = subset(train_idx)
    val_samples = subset(val_idx)
    test_samples = subset(test_idx)

    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = RoomsDataset(train_samples, transform=train_tfms)
    val_ds = RoomsDataset(val_samples, transform=eval_tfms)
    test_ds = RoomsDataset(test_samples, transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = RoomClassifier(num_classes=len(CLASSES), pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_path = OUT_DIR / "models"
    best_path.mkdir(parents=True, exist_ok=True)
    best_model_file = best_path / "room_resnet18_best.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            x, y = x.to(device), torch.tensor(y).to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            tr_correct += (preds == y).sum().item()
            tr_total += x.size(0)

        train_loss = tr_loss / tr_total
        train_acc = tr_correct / tr_total

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for x, y, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                x, y = x.to(device), torch.tensor(y).to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_sum += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "classes": CLASSES, "img_size": IMG_SIZE},
                best_model_file
            )
            print(f"✅ Saved best model: {best_model_file} (val_acc={best_val_acc:.4f})")

    # Test evaluation
    ckpt = torch.load(best_model_file, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y, _ in tqdm(test_loader, desc="[test]"):
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(y)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
