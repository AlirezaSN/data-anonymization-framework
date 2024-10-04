import os
import pandas as pd
import numpy as np

def generate_synthetic_dataset(num_records, num_qids):
    dataset = pd.DataFrame()
    for i in range(num_qids // 2):
        dataset[f"QID_Cat_{i+1}"] = np.random.choice(['A', 'B', 'C', 'D'], size=num_records)
    for i in range(num_qids // 2, num_qids):
        dataset[f"QID_Num_{i+1}"] = np.random.randint(1, 100, size=num_records)
    return dataset

def save_synthetic_datasets(datasets, output_dir="synthetic_datasets"):
    os.makedirs(output_dir, exist_ok=True)
    for num_qids, dataset in datasets.items():
        filename = f"synthetic_dataset_{num_qids}_QIDs.csv"
        file_path = os.path.join(output_dir, filename)
        dataset.to_csv(file_path, index=False)
        print(f"Saved dataset with {num_qids} QIDs to {file_path}")

datasets = {}
num_records = 50000
qids_list = [10, 20, 30, 40, 50]

for num_qids in qids_list:
    datasets[num_qids] = generate_synthetic_dataset(num_records, num_qids)

save_synthetic_datasets(datasets)
