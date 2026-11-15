import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_cleaning import prepare_recordings, FEATURES


# =====================================
# Configuration
# =====================================

WINDOW_SIZE = 256       # 4 seconds at 64 Hz
STEP_SIZE = 128         # 50% overlap
RANDOM_STATE = 42


# =====================================
# 1. Subject-wise Data Splitting
# =====================================

def split_subjects(recordings):

    subjects = sorted({
        str(df["subject"].iloc[0])
        for df in recordings
        if not df.empty
    })

    if len(subjects) < 3:
        raise ValueError(
            "Not enough subjects for train/validation/test split."
        )

    train_subjects, test_subjects = train_test_split(
        subjects,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    train_subjects, val_subjects = train_test_split(
        train_subjects,
        test_size=0.25,
        random_state=RANDOM_STATE
    )

    print("\nTrain subjects:", sorted(train_subjects))
    print("Validation subjects:", sorted(val_subjects))
    print("Test subjects:", sorted(test_subjects))

    return train_subjects, val_subjects, test_subjects


# =====================================
# 2. Select Recordings by Subject
# =====================================

def select_recordings(recordings, subjects):

    subject_set = set(subjects)

    selected = [
        df for df in recordings
        if not df.empty
        and str(df["subject"].iloc[0]) in subject_set
    ]

    return selected


# =====================================
# 3. Fit Normalization on Training Data
# =====================================

def fit_scaler(train_recordings):

    values = [
        df[FEATURES].to_numpy(dtype=np.float32)
        for df in train_recordings
        if not df.empty
    ]

    if not values:
        raise ValueError("No training data available.")

    train_values = np.concatenate(values, axis=0)

    scaler = StandardScaler()
    scaler.fit(train_values)

    return scaler


# =====================================
# 4. Apply Z-score Normalization
# =====================================

def normalize_recordings(recordings, scaler):

    normalized = []

    for df in recordings:

        if df.empty:
            continue

        df = df.copy()

        df.loc[:, FEATURES] = scaler.transform(
            df[FEATURES].to_numpy(dtype=np.float32)
        )

        normalized.append(df)

    return normalized


# =====================================
# 5. Sliding Window Segmentation
# =====================================

def make_windows(recordings):

    X = []
    y = []

    for df in recordings:

        if len(df) < WINDOW_SIZE:
            continue

        values = df[FEATURES].to_numpy(dtype=np.float32)
        labels = df["annotation"].to_numpy()

        # Avoid crossing gaps caused by removed samples
        original_indices = df.index.to_numpy()

        boundaries = np.where(
            np.diff(original_indices) != 1
        )[0] + 1

        segments = np.split(
            np.arange(len(df)),
            boundaries
        )

        for segment in segments:

            if len(segment) < WINDOW_SIZE:
                continue

            for start in range(
                0,
                len(segment) - WINDOW_SIZE + 1,
                STEP_SIZE
            ):

                positions = segment[
                    start:start + WINDOW_SIZE
                ]

                window = values[positions]
                window_labels = labels[positions]

                # Original labels:
                # 1 = Non-FoG
                # 2 = FoG
                #
                # Binary labels:
                # 0 = Non-FoG
                # 1 = FoG

                target = int(
                    np.any(window_labels == 2)
                )

                X.append(window)
                y.append(target)

    if len(X) == 0:
        return (
            np.empty(
                (0, WINDOW_SIZE, len(FEATURES)),
                dtype=np.float32
            ),
            np.empty((0,), dtype=np.int64)
        )

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)

    return X, y


# =====================================
# 6. Main Preprocessing Pipeline
# =====================================

def main():

    print("\nStarting DAPHNET Preprocessing...")

    # Load cleaned and filtered recordings
    recordings = prepare_recordings()

    # Subject-wise split
    train_ids, val_ids, test_ids = split_subjects(
        recordings
    )

    train_rec = select_recordings(
        recordings, train_ids
    )

    val_rec = select_recordings(
        recordings, val_ids
    )

    test_rec = select_recordings(
        recordings, test_ids
    )

    # Fit scaler ONLY on training data
    scaler = fit_scaler(train_rec)

    # Apply normalization
    train_rec = normalize_recordings(
        train_rec, scaler
    )

    val_rec = normalize_recordings(
        val_rec, scaler
    )

    test_rec = normalize_recordings(
        test_rec, scaler
    )

    # Sliding windows
    X_train, y_train = make_windows(train_rec)
    X_val, y_val = make_windows(val_rec)
    X_test, y_test = make_windows(test_rec)

    # Display results
    print("\n========== DATASET SHAPES ==========")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("\n========== LABEL DISTRIBUTION ==========")

    print(
        "Training:",
        np.bincount(y_train, minlength=2)
    )

    print(
        "Validation:",
        np.bincount(y_val, minlength=2)
    )

    print(
        "Testing:",
        np.bincount(y_test, minlength=2)
    )

    print("\nPreprocessing completed successfully!")

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )


# =====================================
# Run Program
# =====================================

if __name__ == "__main__":
    main()


    