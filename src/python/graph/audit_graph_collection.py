import pandas as pd

dataset = pd.read_parquet(DATASET_FILE)

print(dataset.columns.tolist())