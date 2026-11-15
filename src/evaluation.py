
"""
CNN Evaluation for Parkinson's Freezing of Gait Detection.

Expected project structure:
    src/
        cnn_model.py
        evaluation.py
    models/
        best_fog_cnn.pt
    data/
        processed/
            X_test.npy
            y_test.npy

Alternatively:
    data/processed/test.npz

Run:
    python src/evaluation.py
"""

from pathlib import Path
import inspect
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==================================================
# 1. Project Paths and Settings
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_fog_cnn.pt"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"

BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# 2. Load Test Dataset
# ==================================================

def load_test_data():
    print("\n[1/5] Loading test dataset...")

    x_path = DATA_DIR / "X_test.npy"
    y_path = DATA_DIR / "y_test.npy"

    if x_path.exists() and y_path.exists():
        X_test = np.load(x_path, allow_pickle=False)
        y_test = np.load(y_path, allow_pickle=False)

    else:
        npz_path = DATA_DIR / "test.npz"

        if not npz_path.exists():
            raise FileNotFoundError(
                "Test dataset not found.\n"
                f"Expected:\n{x_path}\n{y_path}\n"
                f"or:\n{npz_path}"
            )

        with np.load(npz_path, allow_pickle=False) as data:
            keys = set(data.files)

            if {"X", "y"}.issubset(keys):
                X_test = data["X"]
                y_test = data["y"]

            elif {"X_test", "y_test"}.issubset(keys):
                X_test = data["X_test"]
                y_test = data["y_test"]

            else:
                raise KeyError(
                    f"Unsupported test.npz keys: {sorted(keys)}"
                )

    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test).reshape(-1)

    if X_test.ndim != 3:
        raise ValueError(
            f"Expected X_test shape (N, window, channels), "
            f"got {X_test.shape}"
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test sample counts do not match."
        )

    if not np.isfinite(X_test).all():
        raise ValueError(
            "X_test contains NaN or infinite values."
        )

    if not np.all(np.isin(y_test, [0, 1])):
        raise ValueError(
            "Expected binary labels 0 and 1. "
            f"Found: {np.unique(y_test)}"
        )

    y_test = y_test.astype(np.int64)

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("Label distribution:", np.bincount(y_test, minlength=2))

    return X_test, y_test


# ==================================================
# 3. Locate Model Architecture
# ==================================================

def find_model_class():
    """
    Look for a CNN model class inside src/cnn_model.py.

    This avoids hardcoding a class name that may differ
    between project versions.
    """

    import cnn_model

    preferred_names = [
        "FoGCNN",
        "FOGCNN",
        "CNNModel",
        "CNN1D",
        "CNN1DModel",
        "CNN",
        "FogCNN",
        "FoG_CNN",
    ]

    for name in preferred_names:
        obj = getattr(cnn_model, name, None)

        if (
            inspect.isclass(obj)
            and issubclass(obj, nn.Module)
            and obj is not nn.Module
        ):
            return obj

    candidates = []

    for _, obj in inspect.getmembers(
        cnn_model, inspect.isclass
    ):
        if (
            issubclass(obj, nn.Module)
            and obj is not nn.Module
            and obj.__module__ == cnn_model.__name__
        ):
            candidates.append(obj)

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise RuntimeError(
            "No CNN model class found in src/cnn_model.py. "
            "The evaluation script needs the exact model "
            "architecture used during training."
        )

    raise RuntimeError(
        "Multiple model classes found: "
        + ", ".join(cls.__name__ for cls in candidates)
        + ". Specify the correct trained model class "
        "in find_model_class()."
    )


def create_model(model_class, input_channels):
    """
    Try common constructor argument names.

    If the trained model requires other settings,
    use the exact constructor from cnn_model.py.
    """

    signature = inspect.signature(model_class)
    parameters = signature.parameters

    known_values = {
        "input_channels": input_channels,
        "in_channels": input_channels,
        "num_channels": input_channels,
        "n_channels": input_channels,
        "channels": input_channels,
        "input_dim": input_channels,
        "num_classes": 2,
        "n_classes": 2,
        "output_dim": 2,
    }

    kwargs = {
        name: value
        for name, value in known_values.items()
        if name in parameters
    }

    try:
        return model_class(**kwargs)

    except TypeError as error:
        raise RuntimeError(
            f"Cannot initialize {model_class.__name__} "
            f"with arguments {kwargs}.\n"
            "Use the same model constructor and "
            "hyperparameters as cnn_model.py training.\n"
            f"Original error: {error}"
        ) from error


# ==================================================
# 4. Load Trained CNN Model
# ==================================================

def load_model(input_channels):
    print("\n[2/5] Loading trained CNN model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}\n"
            "Train the CNN model first."
        )

    print("Checkpoint:", MODEL_PATH)
    print("Device:", DEVICE)

    # Load only trusted checkpoints created by your team.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(checkpoint, dict):
        state_dict = None

        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "model_weights",
        ]:
            value = checkpoint.get(key)

            if isinstance(value, dict):
                state_dict = value
                break

        if state_dict is None:
            if checkpoint and all(
                torch.is_tensor(value)
                for value in checkpoint.values()
            ):
                state_dict = checkpoint

        if state_dict is None:
            raise RuntimeError(
                "Checkpoint format not recognized. "
                f"Available keys: {list(checkpoint.keys())}"
            )

    else:
        raise RuntimeError(
            "Expected a PyTorch state_dict checkpoint. "
            f"Found: {type(checkpoint).__name__}"
        )

    model_class = find_model_class()

    print("Model architecture:", model_class.__name__)

    model = create_model(
        model_class,
        input_channels=input_channels,
    )

    try:
        model.load_state_dict(state_dict, strict=True)

    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint weights do not match the "
            "constructed CNN architecture.\n"
            "Check model constructor arguments and "
            "the architecture used during training.\n"
            f"Details:\n{error}"
        ) from error

    model = model.to(DEVICE)
    model.eval()

    print("Model loaded successfully!")

    return model


# ==================================================
# 5. Prepare Test DataLoader
# ==================================================

def create_test_loader(X_test, y_test):
    """
    Convert:
        (N, window, channels)
    into:
        (N, channels, window)

    This is the standard Conv1d input layout.
    """

    X_tensor = torch.from_numpy(
        X_test.transpose(0, 2, 1).copy()
    )

    y_tensor = torch.from_numpy(y_test)

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return loader


# ==================================================
# 6. Run Predictions
# ==================================================

def evaluate_model(model, test_loader):
    print("\n[3/5] Evaluating CNN model...")

    all_predictions = []
    all_labels = []
    all_probabilities = []

    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs = inputs.to(DEVICE)

            outputs = model(inputs)

            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(1)

            if outputs.ndim != 2:
                raise ValueError(
                    "Expected model output shape "
                    "(batch, 1) or (batch, 2). "
                    f"Got {tuple(outputs.shape)}"
                )

            if outputs.shape[1] == 1:
                # Assumes a single binary logit.
                probabilities = torch.sigmoid(
                    outputs[:, 0]
                )

                predictions = (
                    probabilities >= 0.5
                ).long()

            elif outputs.shape[1] == 2:
                # Assumes two class logits.
                probabilities = torch.softmax(
                    outputs,
                    dim=1,
                )[:, 1]

                predictions = torch.argmax(
                    outputs,
                    dim=1,
                )

            else:
                raise ValueError(
                    "Expected 1 or 2 output units. "
                    f"Got {outputs.shape[1]}"
                )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.asarray(all_labels, dtype=int)
    y_pred = np.asarray(all_predictions, dtype=int)
    y_prob = np.asarray(all_probabilities, dtype=float)

    return y_true, y_pred, y_prob


# ==================================================
# 7. Calculate Metrics
# ==================================================

def calculate_metrics(y_true, y_pred, y_prob):
    print("\n[4/5] Calculating evaluation metrics...")

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = float("nan")

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Non-FoG", "FoG"],
        zero_division=0,
    )

    print("\n" + "=" * 48)
    print("       CNN TEST EVALUATION RESULTS")
    print("=" * 48)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ==================================================
# 8. Save Evaluation Results
# ==================================================

def save_results(metrics, y_true, y_pred, y_prob):
    print("\n[5/5] Saving evaluation results...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = OUTPUT_DIR / "evaluation_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write("CNN TEST EVALUATION RESULTS\n")
        file.write("=" * 45 + "\n\n")

        file.write(
            f"Accuracy: {metrics['accuracy']:.4f}\n"
        )

        file.write(
            f"Precision: {metrics['precision']:.4f}\n"
        )

        file.write(
            f"Recall: {metrics['recall']:.4f}\n"
        )

        file.write(
            f"F1-score: {metrics['f1']:.4f}\n"
        )

        file.write(
            f"ROC-AUC: {metrics['roc_auc']:.4f}\n\n"
        )

        file.write("Confusion Matrix:\n")
        file.write(
            str(metrics["confusion_matrix"])
        )

        file.write("\n\nClassification Report:\n")
        file.write(
            metrics["classification_report"]
        )

    predictions_path = (
        OUTPUT_DIR / "test_predictions.csv"
    )

    predictions = np.column_stack(
        [y_true, y_pred, y_prob]
    )

    np.savetxt(
        predictions_path,
        predictions,
        delimiter=",",
        header="true_label,predicted_label,fog_probability",
        comments="",
        fmt=["%d", "%d", "%.6f"],
    )

    cm = metrics["confusion_matrix"]

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-FoG", "FoG"],
    )

    display.plot(
        cmap="Blues",
        values_format="d",
    )

    plt.title("CNN Confusion Matrix")
    plt.tight_layout()

    figure_path = (
        OUTPUT_DIR / "confusion_matrix.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved:", report_path)
    print("Saved:", predictions_path)
    print("Saved:", figure_path)


# ==================================================
# 9. Main Function
# ==================================================

def main():
    print("=" * 55)
    print(" PARKINSON'S DISEASE - CNN FoG EVALUATION")
    print("=" * 55)

    X_test, y_test = load_test_data()

    model = load_model(
        input_channels=X_test.shape[2]
    )

    test_loader = create_test_loader(
        X_test,
        y_test,
    )

    y_true, y_pred, y_prob = evaluate_model(
        model,
        test_loader,
    )

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_prob,
    )

    save_results(
        metrics,
        y_true,
        y_pred,
        y_prob,
    )

    print("\nEvaluation completed successfully!")


# ==================================================
# 10. Run Program
# ==================================================

if __name__ == "__main__":
    main()


# ==================================================
# Additional Evaluation Metric: Balanced Accuracy
# ==================================================

def calculate_balanced_accuracy(y_true, y_pred):
    """
    Calculate balanced accuracy for imbalanced FoG data.
    """
    from sklearn.metrics import balanced_accuracy_score

    score = balanced_accuracy_score(y_true, y_pred)
    return score

