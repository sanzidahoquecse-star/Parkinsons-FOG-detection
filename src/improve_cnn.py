
import copy
import csv
import json
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
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

# Import original CNN architecture
from cnn_model import FoGCNN, load_split


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results" / "cnn_improvement"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
EPOCHS = 20

EXPERIMENTS = [
    {
        "name": "original_settings",
        "method": "class_weights",
        "lr": 0.001,
        "batch_size": 64
    },
    {
        "name": "lower_learning_rate",
        "method": "class_weights",
        "lr": 0.0003,
        "batch_size": 64
    },
    {
        "name": "smaller_batch",
        "method": "class_weights",
        "lr": 0.001,
        "batch_size": 32
    },
    {
        "name": "weighted_sampling",
        "method": "sampling",
        "lr": 0.001,
        "batch_size": 64
    },
    {
        "name": "without_balancing",
        "method": "none",
        "lr": 0.001,
        "batch_size": 64
    }
]


# ==================================================
# 2. REPRODUCIBILITY
# ==================================================

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ==================================================
# 3. CREATE DATA LOADERS
# ==================================================

def create_loaders(train_data, val_data, labels, config):

    counts = np.bincount(labels, minlength=2)

    if np.any(counts == 0):
        raise ValueError(
            "Training data must contain both classes."
        )

    sampler = None

    if config["method"] == "sampling":

        sample_weights = 1.0 / counts[labels]

        sampler = WeightedRandomSampler(
            weights=torch.tensor(
                sample_weights,
                dtype=torch.double
            ),
            num_samples=len(labels),
            replacement=True
        )

    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=False
    )

    if config["method"] == "class_weights":

        weights = len(labels) / (2.0 * counts)

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(
                weights,
                dtype=torch.float32,
                device=DEVICE
            )
        )

    else:
        criterion = nn.CrossEntropyLoss()

    return train_loader, val_loader, criterion


# ==================================================
# 4. TRAINING
# ==================================================

def train_epoch(model, loader, criterion, optimizer):

    model.train()
    total_loss = 0.0

    for X, y in loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


# ==================================================
# 5. GET FoG PROBABILITIES
# ==================================================

def get_probabilities(model, loader):

    model.eval()

    labels = []
    probabilities = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)

            logits = model(X)

            probs = torch.softmax(
                logits, dim=1
            )[:, 1]

            labels.extend(y.numpy().tolist())

            probabilities.extend(
                probs.cpu().numpy().tolist()
            )

    return (
        np.array(labels),
        np.array(probabilities)
    )


# ==================================================
# 6. EVALUATION METRICS
# ==================================================

def calculate_metrics(y_true, probs, threshold):

    predictions = (
        probs >= threshold
    ).astype(int)

    return {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),

        "fog_precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),

        "fog_recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),

        "fog_f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true, predictions
            )
        ),

        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist()
    }


# ==================================================
# 7. VALIDATION THRESHOLD TUNING
# ==================================================

def tune_threshold(y_true, probs):

    best_metrics = None

    for threshold in np.arange(0.10, 0.901, 0.025):

        current = calculate_metrics(
            y_true, probs, threshold
        )

        if best_metrics is None:
            best_metrics = current
            continue

        current_score = (
            current["fog_f1"],
            current["balanced_accuracy"],
            current["fog_recall"]
        )

        best_score = (
            best_metrics["fog_f1"],
            best_metrics["balanced_accuracy"],
            best_metrics["fog_recall"]
        )

        if current_score > best_score:
            best_metrics = current

    return best_metrics


# ==================================================
# 8. SAVE CONFUSION MATRIX
# ==================================================

def save_confusion_matrix(matrix):

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Non-FoG", "FoG"])
    ax.set_yticklabels(["Non-FoG", "FoG"])

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_title("Improved CNN Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(matrix[i][j]),
                ha="center", va="center"
            )

    fig.tight_layout()

    fig.savefig(
        RESULT_DIR / "confusion_matrix.png",
        dpi=150
    )

    plt.close(fig)


# ==================================================
# 9. MAIN PIPELINE
# ==================================================

def main():

    set_seed()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print("Loading DAPHNET datasets...")

    train_data, train_labels = load_split("train")
    val_data, val_labels = load_split("val")

    if len(np.unique(val_labels)) != 2:
        raise ValueError(
            "Validation data must contain both classes."
        )

    print("Training windows:", len(train_data))
    print("Validation windows:", len(val_data))

    all_results = []

    best_result = None
    best_weights = None

    # --------------------------------------------------
    # Train all configurations
    # --------------------------------------------------

    for config in EXPERIMENTS:

        set_seed()

        print("\n===================================")
        print("Experiment:", config["name"])
        print("===================================")

        train_loader, val_loader, criterion = create_loaders(
            train_data,
            val_data,
            train_labels,
            config
        )

        model = FoGCNN().to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"]
        )

        experiment_best = None
        experiment_weights = None
        experiment_best_epoch = None

        for epoch in range(1, EPOCHS + 1):

            loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )

            y_val, val_probs = get_probabilities(
                model, val_loader
            )

            current = tune_threshold(
                y_val, val_probs
            )

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Loss: {loss:.4f} | "
                f"FoG F1: {current['fog_f1']:.4f} | "
                f"Recall: {current['fog_recall']:.4f}"
            )

            current_key = (
                current["fog_f1"],
                current["balanced_accuracy"],
                current["fog_recall"]
            )

            previous_key = None

            if experiment_best is not None:
                previous_key = (
                    experiment_best["fog_f1"],
                    experiment_best["balanced_accuracy"],
                    experiment_best["fog_recall"]
                )

            if previous_key is None or current_key > previous_key:

                experiment_best = current
                experiment_best_epoch = epoch

                experiment_weights = copy.deepcopy(
                    model.state_dict()
                )

        result = {
            "experiment": config["name"],
            "method": config["method"],
            "learning_rate": config["lr"],
            "batch_size": config["batch_size"],
            "best_epoch": experiment_best_epoch,
            **experiment_best
        }

        all_results.append(result)

        print("\nBest validation result:")
        print(result)

        current_key = (
            result["fog_f1"],
            result["balanced_accuracy"],
            result["fog_recall"]
        )

        best_key = None

        if best_result is not None:
            best_key = (
                best_result["fog_f1"],
                best_result["balanced_accuracy"],
                best_result["fog_recall"]
            )

        if best_key is None or current_key > best_key:

            best_result = result
            best_weights = experiment_weights

    # --------------------------------------------------
    # Save validation experiments
    # --------------------------------------------------

    with open(
        RESULT_DIR / "validation_experiments.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(all_results, file, indent=4)

    # --------------------------------------------------
    # Save selected model and threshold
    # --------------------------------------------------

    model_path = MODEL_DIR / "best_fog_cnn_improved.pt"

    torch.save(
        {
            "model_state_dict": best_weights,
            "threshold": best_result["threshold"],
            "selected_configuration": best_result,
            "seed": SEED
        },
        model_path
    )

    print("\nBEST VALIDATION CONFIGURATION:")
    print(best_result)

    # --------------------------------------------------
    # Final test evaluation
    # Test data is loaded only after model selection.
    # --------------------------------------------------

    test_data, _ = load_split("test")

    test_loader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False
    )

    final_model = FoGCNN().to(DEVICE)

    final_model.load_state_dict(best_weights)

    y_test, test_probs = get_probabilities(
        final_model, test_loader
    )

    threshold = best_result["threshold"]

    test_metrics = calculate_metrics(
        y_test,
        test_probs,
        threshold
    )

    test_predictions = (
        test_probs >= threshold
    ).astype(int)

    print("\n========== FINAL TEST RESULTS ==========")

    for key, value in test_metrics.items():
        print(key, ":", value)

    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Non-FoG", "FoG"],
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # Save final test results
    with open(
        RESULT_DIR / "final_test_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "selected_configuration": best_result,
                "test_metrics": test_metrics
            },
            file,
            indent=4
        )

    with open(
        RESULT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    with open(
        RESULT_DIR / "test_predictions.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "true_label",
            "fog_probability",
            "predicted_label"
        ])

        writer.writerows(
            zip(
                y_test.tolist(),
                test_probs.tolist(),
                test_predictions.tolist()
            )
        )

    save_confusion_matrix(
        test_metrics["confusion_matrix"]
    )

    print("\nSTEP 1 COMPLETED SUCCESSFULLY!")
    print("Model:", model_path)
    print("Results:", RESULT_DIR)


if __name__ == "__main__":
    main()
