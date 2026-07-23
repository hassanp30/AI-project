import io
from pathlib import Path
import pandas as pd

# Paths relative to this file's location
BASE_DIR = Path(__file__).resolve().parent
RAW_DATASET_PATH = BASE_DIR / "dataset" / "WISDM_ar_v1.1_raw.txt"
CLEAN_DATASET_PATH = BASE_DIR / "dataset" / "WISDM_ar_v1.1_cleaned.csv"


def load_clean_dataset(raw_path: Path = RAW_DATASET_PATH) -> pd.DataFrame:
    """
    Reads the raw WISDM v1.1 dataset, cleans formatting artifacts 
    (missing newlines, trailing ',;', empty lines), and returns a clean DataFrame.
    """
    clean_lines = []

    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split on ';' to handle records missing newline breaks
            sub_lines = line.split(";")
            for sub in sub_lines:
                sub = sub.strip()
                if not sub:
                    continue

                # Remove trailing comma if present (e.g., ',;' anomaly)
                if sub.endswith(","):
                    sub = sub[:-1]

                clean_lines.append(sub)

    # Convert cleaned string lines into Pandas DataFrame
    clean_csv_data = "\n".join(clean_lines)
    df = pd.read_csv(
        io.StringIO(clean_csv_data),
        header=None,
        names=["user", "activity", "timestamp", "x", "y", "z"],
        on_bad_lines="skip",
        low_memory=False,
    )

    # Ensure numeric columns are properly typed
    df["user"] = pd.to_numeric(df["user"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")

    # Drop any remaining NaN rows caused by corrupted string values
    df = df.dropna().reset_index(drop=True)

    # Convert user integer ID type
    df["user"] = df["user"].astype(int)

    return df


def clean_and_save_csv(raw_path: Path = RAW_DATASET_PATH, output_path: Path = CLEAN_DATASET_PATH) -> pd.DataFrame:
    """
    Cleans the raw dataset and saves the result to a clean CSV file.
    """
    print(f"Loading and cleaning raw dataset from: {raw_path}...")
    df = load_clean_dataset(raw_path)

    print(f"Saving cleaned dataset to: {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df):,} cleaned rows!")
    return df


if __name__ == "__main__":
    df = clean_and_save_csv()
    print("\nDataset Preview:")
    print(df.head())
    print("\nClass Distribution:")
    print(df["activity"].value_counts())
