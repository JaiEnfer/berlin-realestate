# train_multitask_tb.py
# Multi-task training (Room 5-way + Quality binary) with TensorBoard logging.
#
# Logs:
# - train/loss_room, train/loss_quality, train/loss_total
# - train/room_acc (running per epoch end)
# - val/room_acc, val/loss_room, val/loss_quality, val/loss_total
# - test/quality ROC-AUC, PR-AUC, best threshold + metrics @ best threshold
#
# Usage (from project root):
#   cd src
#   python train_multitask_tb.py
#
# TensorBoard:
#   tensorboard --logdir outputs/logs

from pathlib import Path
import random
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from tqdm import tqdm

from dataset_multitask import MultiTaskRoomsDataset
from model_multitask import MultiTaskResNet18


# -------------------------
# Config
# -------------------------
CLASSES = ["Bathroom", "Bedroom", "Dinning", "Kitchen", "Livingroom"]
CSV_PATH = "data/processed/quality.csv"

OUT_DIR = Path("outputs")
MODELS_DIR = OUT_DIR / "models"
LOGS_DIR = OUT_DIR / "logs" / "multitask"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4
IMG_SIZE = 224

# Tune this to trade off room vs quality learning
LAMBDA_QUALITY = 0.7

# Best checkpointing uses a combined validation score
# combined = val_room_acc - 0.10 * val_q_loss
VAL_Q_LOSS_WEIGHT_IN_SCORE = 0.10


# -------------------------
# Helpers
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_best_threshold(y_true, y_prob):
    """Find threshold that maximizes F1."""
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)

    best = {"threshold": 0.5, "f1": -1, "precision": 0, "recall": 0}
    for t in np.linspace(0.05, 0.95, 19):
        y_hat = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        p = precision_score(y_true, y_hat, zero_division=0)
        r = recall_score(y_true, y_hat, zero_division=0)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1), "precision": float(p), "recall": float(r)}
    return best


def accuracy_from_logits(logits, y_true):
    preds = logits.argmax(dim=1)
    return float((preds == y_true).float().mean().item())


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Make log run name unique per run
    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(LOGS_DIR / run_name))
    writer.add_text(
        "config",
        f"SEED={SEED}, BATCH={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LR}, IMG_SIZE={IMG_SIZE}, "
        f"LAMBDA_QUALITY={LAMBDA_QUALITY}, VAL_Q_LOSS_WEIGHT_IN_SCORE={VAL_Q_LOSS_WEIGHT_IN_SCORE}",
    )

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["path", "room", "quality_label"]).reset_index(drop=True)

    idx = np.arange(len(df))
    room_labels = df["room"].values

    # 70/15/15 stratified by room
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, random_state=SEED, stratify=room_labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=room_labels[temp_idx]
    )

    # Transforms
    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.5
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = MultiTaskRoomsDataset(CSV_PATH, CLASSES, transform=train_tfms, indices=train_idx)
    val_ds = MultiTaskRoomsDataset(CSV_PATH, CLASSES, transform=eval_tfms, indices=val_idx)
    test_ds = MultiTaskRoomsDataset(CSV_PATH, CLASSES, transform=eval_tfms, indices=test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # pos_weight for quality imbalance
    q_labels_train = df.iloc[train_idx]["quality_label"].astype(int).values
    pos = int((q_labels_train == 1).sum())
    neg = int((q_labels_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)
    print(f"Quality train distribution: pos={pos}, neg={neg}, pos_weight={pos_weight.item():.3f}")
    writer.add_scalar("data/quality_pos", pos, 0)
    writer.add_scalar("data/quality_neg", neg, 0)
    writer.add_scalar("data/pos_weight", float(pos_weight.item()), 0)

    # Model + losses + optimizer
    model = MultiTaskResNet18(num_classes=len(CLASSES), pretrained=True).to(device)
    room_criterion = nn.CrossEntropyLoss()
    quality_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Checkpointing
    best_val = -1e9
    best_file = MODELS_DIR / "multitask_resnet18_best_tb.pt"

    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        # -------------------------
        # TRAIN
        # -------------------------
        model.train()

        tr_total = 0
        tr_room_correct = 0
        tr_room_loss_sum = 0.0
        tr_q_loss_sum = 0.0
        tr_total_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for x, y_room, y_q, _ in pbar:
            x = x.to(device)
            y_room = torch.tensor(y_room).to(device)
            y_q = torch.tensor(y_q, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            room_logits, q_logit = model(x)

            loss_room = room_criterion(room_logits, y_room)
            loss_q = quality_criterion(q_logit, y_q)
            loss_total = loss_room + LAMBDA_QUALITY * loss_q

            loss_total.backward()
            optimizer.step()

            bs = x.size(0)
            tr_room_loss_sum += loss_room.item() * bs
            tr_q_loss_sum += loss_q.item() * bs
            tr_total_loss_sum += loss_total.item() * bs

            preds_room = room_logits.argmax(dim=1)
            tr_room_correct += (preds_room == y_room).sum().item()
            tr_total += bs

            # TensorBoard per-step (batch)
            writer.add_scalar("train/loss_room_step", loss_room.item(), global_step)
            writer.add_scalar("train/loss_quality_step", loss_q.item(), global_step)
            writer.add_scalar("train/loss_total_step", loss_total.item(), global_step)
            global_step += 1

            pbar.set_postfix(loss=float(loss_total.item()))

        train_room_acc = tr_room_correct / tr_total
        train_room_loss = tr_room_loss_sum / tr_total
        train_q_loss = tr_q_loss_sum / tr_total
        train_total_loss = tr_total_loss_sum / tr_total

        writer.add_scalar("train/room_acc", train_room_acc, epoch)
        writer.add_scalar("train/loss_room", train_room_loss, epoch)
        writer.add_scalar("train/loss_quality", train_q_loss, epoch)
        writer.add_scalar("train/loss_total", train_total_loss, epoch)

        # -------------------------
        # VAL
        # -------------------------
        model.eval()

        val_total = 0
        val_room_correct = 0
        val_room_loss_sum = 0.0
        val_q_loss_sum = 0.0
        val_total_loss_sum = 0.0

        with torch.no_grad():
            for x, y_room, y_q, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                x = x.to(device)
                y_room = torch.tensor(y_room).to(device)
                y_q = torch.tensor(y_q, dtype=torch.float32).to(device)

                room_logits, q_logit = model(x)

                loss_room = room_criterion(room_logits, y_room)
                loss_q = quality_criterion(q_logit, y_q)
                loss_total = loss_room + LAMBDA_QUALITY * loss_q

                bs = x.size(0)
                val_room_loss_sum += loss_room.item() * bs
                val_q_loss_sum += loss_q.item() * bs
                val_total_loss_sum += loss_total.item() * bs

                preds_room = room_logits.argmax(dim=1)
                val_room_correct += (preds_room == y_room).sum().item()
                val_total += bs

        val_room_acc = val_room_correct / val_total
        val_room_loss = val_room_loss_sum / val_total
        val_q_loss = val_q_loss_sum / val_total
        val_total_loss = val_total_loss_sum / val_total

        writer.add_scalar("val/room_acc", val_room_acc, epoch)
        writer.add_scalar("val/loss_room", val_room_loss, epoch)
        writer.add_scalar("val/loss_quality", val_q_loss, epoch)
        writer.add_scalar("val/loss_total", val_total_loss, epoch)

        # Combined score for checkpointing
        combined = val_room_acc - VAL_Q_LOSS_WEIGHT_IN_SCORE * val_q_loss
        writer.add_scalar("val/combined_score", combined, epoch)

        print(
            f"\nEpoch {epoch}: "
            f"train_room_acc={train_room_acc:.4f} val_room_acc={val_room_acc:.4f} | "
            f"train_total_loss={train_total_loss:.4f} val_total_loss={val_total_loss:.4f} | "
            f"train_q_loss={train_q_loss:.4f} val_q_loss={val_q_loss:.4f} | "
            f"combined={combined:.4f}"
        )

        if combined > best_val:
            best_val = combined
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": CLASSES,
                    "img_size": IMG_SIZE,
                    "lambda_quality": LAMBDA_QUALITY,
                    "pos_weight": float(pos_weight.item()),
                    "seed": SEED,
                    "run_name": run_name,
                },
                best_file,
            )
            print(f"✅ Saved best model: {best_file}")

    # -------------------------
    # TEST EVAL (best checkpoint)
    # -------------------------
    ckpt = torch.load(best_file, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred = [], []
    q_true, q_prob_all = [], []

    with torch.no_grad():
        for x, y_room, y_q, _ in tqdm(test_loader, desc="[test]"):
            x = x.to(device)
            room_logits, q_logit = model(x)

            preds_room = room_logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds_room)
            y_true.extend([int(v) for v in y_room])

            q_prob = torch.sigmoid(q_logit).cpu().numpy().tolist()
            q_prob_all.extend(q_prob)
            q_true.extend([int(v) for v in y_q])

    print("\nROOM classification report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    print("ROOM confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Quality probabilistic metrics
    if len(set(q_true)) > 1:
        roc = roc_auc_score(q_true, q_prob_all)
        ap = average_precision_score(q_true, q_prob_all)
        print("\nQUALITY metrics (probabilistic):")
        print(f"ROC-AUC: {roc:.4f}")
        print(f"PR-AUC:  {ap:.4f}")

        writer.add_scalar("test/quality_roc_auc", roc, 0)
        writer.add_scalar("test/quality_pr_auc", ap, 0)
    else:
        print("\nQUALITY metrics: only one class present in y_true; ROC/PR AUC undefined.")

    best = find_best_threshold(q_true, q_prob_all)
    print("\nQUALITY best threshold (max F1):")
    print(best)

    t = best["threshold"]
    q_hat = (np.array(q_prob_all) >= t).astype(int)
    acc = float((q_hat == np.array(q_true)).mean())
    p = precision_score(q_true, q_hat, zero_division=0)
    r = recall_score(q_true, q_hat, zero_division=0)
    f1 = f1_score(q_true, q_hat, zero_division=0)

    print(f"\nQUALITY metrics @ threshold={t:.2f}: acc={acc:.4f} precision={p:.4f} recall={r:.4f} f1={f1:.4f}")

    writer.add_scalar("test/quality_best_threshold", float(t), 0)
    writer.add_scalar("test/quality_acc_at_best_t", acc, 0)
    writer.add_scalar("test/quality_precision_at_best_t", float(p), 0)
    writer.add_scalar("test/quality_recall_at_best_t", float(r), 0)
    writer.add_scalar("test/quality_f1_at_best_t", float(f1), 0)

    writer.close()
    print(f"\n✅ TensorBoard logs saved to: {LOGS_DIR / run_name}")
    print("Run: tensorboard --logdir outputs/logs")


if __name__ == "__main__":
    main()
