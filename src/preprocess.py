from pathlib import Path

import pandas as pd

from preprocess_imu import preprocess_imu
from preprocess_tof import preprocess_tof

KEYS = ["sequence_id", "sequence_counter"]
# gesture and sequence_type only exist in train
INFO_COLS = ["subject", "gesture", "sequence_type"]


def build_dataset(df, demo):
    """Run IMU and ToF/thermal preprocessing and merge the results.

    Input:
        df   : raw sensor data (train or test), one row per time step
        demo : demographics table, one row per subject
    Output:
        DataFrame with the same number of rows as df: keys, info columns
        (only those present in df), IMU columns, ToF/thermal columns.
    """
    imu = preprocess_imu(df, demo)
    tof = preprocess_tof(df, demo)

    # Column names from the two parts must not overlap (except the keys)
    common = (set(imu.columns) & set(tof.columns)) - set(KEYS)
    assert not common, f"Duplicate columns: {common}"

    # Keep only the info columns that exist (the test set has no labels)
    info = [c for c in INFO_COLS if c in df.columns]

    data = df[KEYS + info].merge(imu, on=KEYS).merge(tof, on=KEYS)

    assert len(data) == len(df), "Number of rows changed after merging"
    return data


if __name__ == "__main__":
    train = pd.read_csv("data/raw/train.csv")
    demo = pd.read_csv("data/raw/train_demographics.csv")

    data = build_dataset(train, demo)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    data.to_parquet("data/processed/train.parquet", index=False)

    print("Rows:", len(data))
    print("Columns:", data.shape[1])
    print(data.head())
