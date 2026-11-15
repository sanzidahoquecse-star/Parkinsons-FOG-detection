
"""
1D CNN Baseline for Parkinson's Freezing of Gait Detection.

Expected preprocessing output:
    data/processed/train.npz
    data/processed/val.npz
    data/processed/test.npz

Each NPZ file must contain:
    X: shape (number_of_windows, window_length, 3)
    y: shape (number_of_windows,)

Labels:
    0 = Non-FoG
    1 = FoG
"""

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

SEED = 42
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. DATA LOADING
# ============================================================

def load_split(split_name):
    file_path = DATA_DIR / f"{split_name}.npz"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing file: {file_path}\n"
            "Run preprocessing first and export "
            "train.npz, val.npz, and test.npz."
        )

    with np.load(file_path) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)

    if X.ndim != 3 or X.shape[2] != 3:
        raise ValueError(
            f"{split_name}: expected X shape (N, T, 3), "
            f"got {X.shape}"
        )

    if y.ndim != 1 or len(y) != len(X):
        raise ValueError(
            f"{split_name}: X and y have incompatible shapes."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{split_name}: X contains NaN or infinity."
        )

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            f"{split_name}: labels must be 0 or 1. "
            "Convert original DAPHNET labels "
            "1 -> 0 and 2 -> 1 during preprocessing."
        )

    # PyTorch Conv1d expects (N, channels, time).
    X = np.transpose(X, (0, 2, 1))

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    return TensorDataset(X_tensor, y_tensor), y


# ============================================================
# 4. 1D CNN MODEL
# ============================================================

class FoGCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return logits


# ============================================================
# 5. TRAINING AND VALIDATION
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            if is_training:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y_batch)

            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )
            all_labels.extend(
                y_batch.cpu().numpy().tolist()
            )

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    return average_loss, accuracy, f1


# ============================================================
# 6. TEST EVALUATION
# ============================================================

def evaluate_model(model, test_loader):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)

            logits = model(X_batch)
            predictions = torch.argmax(logits, dim=1)

            y_true.extend(y_batch.numpy().tolist())
            y_pred.extend(
                predictions.cpu().numpy().tolist()
            )

    print("\n========== TEST RESULTS ==========")

    print(
        "Accuracy:",
        round(accuracy_score(y_true, y_pred), 4),
    )

    print(
        "Precision:",
        round(
            precision_score(
                y_true, y_pred, zero_division=0
            ),
            4,
        ),
    )

    print(
        "Recall:",
        round(
            recall_score(
                y_true, y_pred, zero_division=0
            ),
            4,
        ),
    )

    print(
        "F1-score:",
        round(
            f1_score(
                y_true, y_pred, zero_division=0
            ),
            4,
        ),
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Non-FoG", "FoG"],
            zero_division=0,
        )
    )


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print("Loading preprocessed datasets...")

    train_dataset, train_labels = load_split("train")
    val_dataset, _ = load_split("val")
    test_dataset, _ = load_split("test")

    print("Training windows:", len(train_dataset))
    print("Validation windows:", len(val_dataset))
    print("Testing windows:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = FoGCNN().to(DEVICE)

    # Handle imbalanced FoG and Non-FoG classes.
    class_counts = np.bincount(
        train_labels,
        minlength=2,
    )

    if np.any(class_counts == 0):
        raise ValueError(
            "Training data must contain both "
            "Non-FoG and FoG windows."
        )

    class_weights = (
        len(train_labels) /
        (2.0 * class_counts)
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_f1 = -1.0
    model_path = MODEL_DIR / "best_fog_cnn.pt"

    print("\nStarting CNN training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_acc, val_f1 = run_epoch(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(),
                model_path,
            )

    print("\nBest validation F1:", round(best_val_f1, 4))
    print("Model saved at:", model_path)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    evaluate_model(model, test_loader)

    print("\nCNN pipeline completed successfully!")


if __name__ == "__main__":
    main()
