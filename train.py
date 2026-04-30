"""
Fruit Classifier — Training Script
Uses MobileNetV2 with pretrained ImageNet weights (transfer learning).

Expected folder structure:
    data/
        train/
            freshapples/    (images)
            freshbanana/
            freshoranges/
            rottenapples/
            rottenbanana/
            rottenoranges/
        val/
            freshapples/
            ...

Run:
    python train.py
"""

import json
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models, datasets
from torchvision.models import MobileNet_V2_Weights
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(r"C:\fruits_detection\dataset")
SAVE_MODEL   = Path("fruit_classifier.pth")
SAVE_CLASSES = Path("class_names.json")

IMG_SIZE     = 128
BATCH_SIZE   = 32
EPOCHS_HEAD  = 5       # train only classifier head (backbone frozen)
EPOCHS_FULL  = 15      # fine-tune entire network
LR_HEAD      = 1e-3
LR_FULL      = 1e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Datasets ──────────────────────────────────────────────────────────────────
train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
val_dataset   = datasets.ImageFolder(DATA_DIR / "val",   transform=val_transform)

CLASS_NAMES = train_dataset.classes
print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

# Save class names
with open(SAVE_CLASSES, "w") as f:
    json.dump(CLASS_NAMES, f)
print(f"Saved class names → {SAVE_CLASSES}")

# ── Weighted sampler (handles class imbalance) ────────────────────────────────
counts       = np.array([sum(1 for _, l in train_dataset.samples if l == i)
                         for i in range(len(CLASS_NAMES))])
print(f"Samples per class: { {CLASS_NAMES[i]: counts[i] for i in range(len(CLASS_NAMES))} }")

class_weights   = 1.0 / counts
sample_weights  = [class_weights[label] for _, label in train_dataset.samples]
sampler         = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)  # pretrained!
model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights / class_weights.sum(), dtype=torch.float).to(DEVICE)
)

# ── Training helper ───────────────────────────────────────────────────────────
def run_epoch(loader, training):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


def evaluate_full():
    """Per-class accuracy report + confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            preds = model(imgs.to(DEVICE)).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\n── Per-class Report ─────────────────────────────")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cmap="Blues")
    plt.title("Confusion Matrix (Validation)")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")


# ── Phase 1: Train head only (backbone frozen) ────────────────────────────────
print(f"\n{'='*55}")
print(f" PHASE 1 — Head only ({EPOCHS_HEAD} epochs, backbone frozen)")
print(f"{'='*55}")

for param in model.features.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR_HEAD)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

best_val_acc  = 0.0
best_weights  = None

for epoch in range(1, EPOCHS_HEAD + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, training=True)
    vl_loss, vl_acc = run_epoch(val_loader,   training=False)
    scheduler.step()

    marker = " ◄ best" if vl_acc > best_val_acc else ""
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        best_weights = copy.deepcopy(model.state_dict())

    print(f"Epoch {epoch:02d}/{EPOCHS_HEAD}  "
          f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}  "
          f"val_loss={vl_loss:.4f} val_acc={vl_acc:.3f}  "
          f"({time.time()-t0:.1f}s){marker}")

# ── Phase 2: Fine-tune entire network ─────────────────────────────────────────
print(f"\n{'='*55}")
print(f" PHASE 2 — Full fine-tune ({EPOCHS_FULL} epochs, all layers)")
print(f"{'='*55}")

for param in model.features.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=LR_FULL)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FULL)

for epoch in range(1, EPOCHS_FULL + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, training=True)
    vl_loss, vl_acc = run_epoch(val_loader,   training=False)
    scheduler.step()

    marker = " ◄ best" if vl_acc > best_val_acc else ""
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        best_weights = copy.deepcopy(model.state_dict())

    print(f"Epoch {epoch:02d}/{EPOCHS_FULL}  "
          f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}  "
          f"val_loss={vl_loss:.4f} val_acc={vl_acc:.3f}  "
          f"({time.time()-t0:.1f}s){marker}")

# ── Save best model ───────────────────────────────────────────────────────────
model.load_state_dict(best_weights)
torch.save(best_weights, SAVE_MODEL)
print(f"\nBest val acc: {best_val_acc:.3f}")
print(f"Saved model → {SAVE_MODEL}")

# ── Final evaluation ──────────────────────────────────────────────────────────
evaluate_full()
print("\nTraining complete!")