
from pathlib import Path
import re
import zipfile
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

URL = (
    "https://archive.ics.uci.edu/static/public/245/"
    "daphnet+freezing+of+gait.zip"
)

COLUMNS = [
    "time",
    "shank_x", "shank_y", "shank_z",
    "thigh_x", "thigh_y", "thigh_z",
    "trunk_x", "trunk_y", "trunk_z",
    "annotation"
]


def download_dataset():
    zip_path = DATA_DIR / "daphnet.zip"

    if not zip_path.exists():
        print("Downloading DAPHNET...")
        response = requests.get(URL, timeout=180)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

    # Extract outer and nested ZIP files
    extracted = set()

    while True:
        archives = [
            p for p in DATA_DIR.rglob("*.zip")
            if p not in extracted
        ]

        if not archives:
            break

        for archive_path in archives:
            with zipfile.ZipFile(archive_path) as z:
                z.extractall(archive_path.parent)

            extracted.add(archive_path)

    print("Dataset ready!")


def load_recordings():
    download_dataset()

    recordings = []

    files = sorted(
        p for p in DATA_DIR.rglob("*.txt")
        if re.fullmatch(
            r"S\d+R\d+", p.stem, re.IGNORECASE
        )
    )

    if not files:
        raise FileNotFoundError(
            "No DAPHNET recording files found."
        )

    for file in files:
        df = pd.read_csv(
            file,
            sep=r"\s+",
            header=None,
            names=COLUMNS
        )

        df["subject"] = re.match(
            r"S\d+", file.stem, re.IGNORECASE
        ).group().upper()

        df["recording"] = file.stem

        recordings.append(df)

    print("Total recordings:", len(recordings))

    return recordings


if __name__ == "__main__":
    recordings = load_recordings()

    first = recordings[0]

    print("\nFirst five rows:")
    print(first.head())

    print("\nDataset shape:", first.shape)

    print("\nAnnotation counts:")
    print(first["annotation"].value_counts())

    print("\nMissing values:")
    print(first.isna().sum())
