
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)

from cnn_model import FoGCNN, load_split


# Configuration
ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results" / "cnn_step2"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEEDS = [42, 123]
METHODS = ["class_weights", "weighted_sampling"]

EPOCHS = 20
BATCH_SIZE = 64
LR = 0.001
THRESHOLD = 0.5


# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Create training loader
def create_loader(dataset, labels, method):

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(labels, minlength=2)

    if len(counts) != 2 or np.any(counts == 0):
        raise ValueError("Both binary classes are required.")

    if method == "class_weights":

        weights = len(labels) / (2.0 * counts)

        weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=DEVICE
        )

        criterion = nn.CrossEntropyLoss(weight=weights)

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

    elif method == "weighted_sampling":

        sample_weights = 1.0 / counts[labels]

        sampler = WeightedRandomSampler(
            weights=torch.tensor(
                sample_weights,
                dtype=torch.double
            ),
            num_samples=len(labels),
            replacement=True
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler
        )

        criterion = nn.CrossEntropyLoss()

    else:
        raise ValueError("Unknown training method.")

    return loader, criterion


# Train one epoch
def train_epoch(model, loader, criterion, optimizer):

    model.train()
    total_loss = 0.0

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE).long().reshape(-1)

        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


# Validation
def validate(model, loader):

    model.eval()

    actual = []
    predicted = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)

            outputs = model(x)

            probabilities = torch.softmax(
                outputs, dim=1
            )[:, 1]

            predictions = (
                probabilities >= THRESHOLD
            ).long()

            actual.extend(y.reshape(-1).cpu().tolist())
            predicted.extend(predictions.cpu().tolist())

    return {
        "accuracy": accuracy_score(actual, predicted),

        "precision": precision_score(
            actual, predicted, zero_division=0
        ),

        "recall": recall_score(
            actual, predicted, zero_division=0
        ),

        "f1": f1_score(
            actual, predicted, zero_division=0
        ),

        "balanced_accuracy": balanced_accuracy_score(
            actual, predicted
        )
    }


# Run individual experiment
def run_experiment(
    method,
    seed,
    train_dataset,
    train_labels,
    val_dataset
):

    set_seed(seed)

    train_loader, criterion = create_loader(
        train_dataset,
        train_labels,
        method
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = FoGCNN().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    best_result = None
    history = []

    print("\n" + "=" * 60)
    print(f"Method: {method} | Seed: {seed}")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):

        loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        metrics = validate(model, val_loader)

        result = {
            "method": method,
            "seed": seed,
            "epoch": epoch,
            "threshold": THRESHOLD,
            "train_loss": loss,
            **metrics
        }

        history.append(result)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

        if (
            best_result is None
            or result["f1"] > best_result["f1"]
        ):
            best_result = result.copy()

    return best_result, history


# Save CSV results
def save_csv(filename, rows):

    path = RESULT_DIR / filename

    with open(path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


# Main program
def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("CNN Stability and Validation Experiment")
    print("Device:", DEVICE)

    train_dataset, train_labels = load_split("train")
    val_dataset, val_labels = load_split("val")

    if set(np.asarray(val_labels).reshape(-1)) != {0, 1}:
        raise ValueError(
            "Validation requires both binary classes."
        )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    results = []
    all_history = []

    for method in METHODS:

        for seed in SEEDS:

            best_result, history = run_experiment(
                method,
                seed,
                train_dataset,
                train_labels,
                val_dataset
            )

            results.append(best_result)
            all_history.extend(history)

    save_csv("validation_results.csv", results)
    save_csv("epoch_history.csv", all_history)

    summary = []

    print("\nFINAL VALIDATION SUMMARY")
    print("=" * 60)

    for method in METHODS:

        selected = [
            r for r in results
            if r["method"] == method
        ]

        row = {
            "method": method,
            "runs": len(selected)
        }

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy"
        ]:

            values = [
                r[metric] for r in selected
            ]

            row[f"{metric}_mean"] = float(
                np.mean(values)
            )

            row[f"{metric}_std"] = float(
                np.std(values, ddof=1)
            )

        summary.append(row)

        print(f"\nMethod: {method}")

        print(
            f"Mean Accuracy: "
            f"{row['accuracy_mean'] * 100:.2f}%"
        )

        print(
            f"Mean Recall: "
            f"{row['recall_mean'] * 100:.2f}%"
        )

        print(
            f"Mean F1-score: "
            f"{row['f1_mean'] * 100:.2f}%"
        )

        print(
            f"F1 Standard Deviation: "
            f"{row['f1_std']:.4f}"
        )

    save_csv("method_summary.csv", summary)

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
