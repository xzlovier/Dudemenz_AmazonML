import pandas as pd
import os

data_dir = os.path.join("student_resource", "dataset")

files_to_check = {
    "train_ground_truth": os.path.join(data_dir, "train", "train_ground_truth.tsv"),
    "train_source1": os.path.join(data_dir, "train", "train_source1.tsv"),
    "train_source2": os.path.join(data_dir, "train", "train_source2.tsv"),
    "train_source3": os.path.join(data_dir, "train", "train_source3.tsv"),
    "test_source1": os.path.join(data_dir, "test", "test_source1.tsv"),
    "test_source2": os.path.join(data_dir, "test", "test_source2.tsv"),
    "test_source3": os.path.join(data_dir, "test", "test_source3.tsv"),
}

for name, path in files_to_check.items():
    print(f"Loading {name} from {path}...")
    df = pd.read_csv(path, sep="\t", dtype=str)
    print(f"Shape: {df.shape}")
    print("Dtypes:")
    print(df.dtypes)
    print(f"Columns: {list(df.columns)}")
    print("-" * 50)
