import numpy as np
from scipy.signal import butter, sosfiltfilt

from data_loading import load_recordings

FEATURES = ["shank_x", "shank_y", "shank_z"]

SAMPLING_RATE = 64
CUTOFF = 20


def clean_recording(df):
    df = df.copy()

    # Convert sensor values to numeric
    for col in FEATURES:
        df[col] = df[col].replace(
            [np.inf, -np.inf], np.nan
        )
        df[col] = df[col].astype(float)

    # Keep only valid labels: 1 and 2
    df = df[df["annotation"].isin([1, 2])]

    # Remove rows with missing sensor values
    df = df.dropna(subset=FEATURES)

    return df


def remove_noise(df):
    df = df.copy()

    sos = butter(
        4,
        CUTOFF,
        btype="lowpass",
        fs=SAMPLING_RATE,
        output="sos"
    )

    # Find gaps in the original sample indices
    indices = df.index.to_numpy()

    boundaries = np.where(
        np.diff(indices) != 1
    )[0] + 1

    segments = np.split(
        np.arange(len(df)), boundaries
    )

    for segment in segments:
        if len(segment) < 20:
            continue

        positions = df.index[segment]

        for col in FEATURES:
            values = df.loc[
                positions, col
            ].to_numpy(dtype=float)

            filtered = sosfiltfilt(sos, values)

            df.loc[positions, col] = filtered

    return df


def prepare_recordings():
    raw_recordings = load_recordings()
    processed = []

    for df in raw_recordings:
        cleaned = clean_recording(df)
        filtered = remove_noise(cleaned)

        processed.append(filtered)

    print("Cleaned recordings:", len(processed))

    return processed


if __name__ == "__main__":
    recordings = prepare_recordings()

    print("\nCleaned sample:")
    print(
        recordings[0][
            FEATURES + ["annotation"]
        ].head()
    )

    print("\nMissing values:")
    print(
        recordings[0][FEATURES].isna().sum()
    )

    