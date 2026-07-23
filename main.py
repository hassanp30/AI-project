import numpy as np
import pandas as pd
from pathlib import Path

FILE_PATH = Path(__file__).resolve().parent

def main():
    dataset_path = Path.resolve(FILE_PATH / ".." / "project" / "dataset" / "WISDM_ar_v1.1_cleaned.csv")
    dataset_file = pd.read_csv(dataset_path,
        header=None,
        low_memory = False,
        names = ['user', 'activity', 'timestamp', 'x', 'y', 'z'],
        sep=","
    )

    train_data = dataset_file[(dataset_file['user']) <= 25]
    test_data = dataset_file[dataset_file['user'] > 25]
    print(test_data.head())


if __name__ == "__main__":
    main()