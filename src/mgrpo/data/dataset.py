import random
import numpy as np
import torch
from datasets import load_dataset
from mgrpo.config import DATA_PATH, DATASET_FRACTION, TRAIN_FRACTION

def get_user_prompt(item):
    for key in ("prompt", "text", "instruction", "user"):
        if key in item and item[key] is not None and str(item[key]).strip():
            return str(item[key]).strip()
    raise KeyError(f"Could not find a prompt field in row keys: {list(item.keys())}")

def load_data(seed: int):
    if not DATA_PATH.exists():
        # In smoke-test mode or when data is missing, we create a dummy dataset
        from datasets import Dataset
        dummy_data = [{"prompt": f"Dummy prompt {i} for testing."} for i in range(100)]
        full_ds = Dataset.from_list(dummy_data)
        print(f"Warning: Missing {DATA_PATH}. Using dummy data for testing.")
    else:
        full_ds = load_dataset("json", data_files=str(DATA_PATH))["train"]
    
    total_all = len(full_ds)
    n_use = max(1, int(total_all * DATASET_FRACTION))
    dataset = full_ds.shuffle(seed=seed).select(range(n_use))

    split_idx = max(1, int(len(dataset) * TRAIN_FRACTION))
    train_ds = dataset.select(range(split_idx))
    val_ds = dataset.select(range(split_idx, len(dataset))) if split_idx < len(dataset) else dataset.select([])

    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} validation from {total_all} total rows")
    return train_ds, val_ds
