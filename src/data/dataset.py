from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FloodDataset(Dataset):
    """PyTorch dataset for preprocessed Sen1Floods11 samples."""

    SPLIT_MAPPING = {
        "train": "train_data",
        "valid": "valid_data",
        "test": "test_data",
        "bolivia": "bolivia_data",
    }

    def __init__(self, split, processed_dir="data/processed"):
        if split not in self.SPLIT_MAPPING:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Choose from {list(self.SPLIT_MAPPING)}"
            )

        self.processed_dir = Path(processed_dir)
        summary_path = self.processed_dir / "preprocessing_summary.csv"

        if not summary_path.exists():
            raise FileNotFoundError(summary_path)

        summary = pd.read_csv(summary_path)

        actual_split = self.SPLIT_MAPPING[split]

        self.samples = summary[
            summary["split"] == actual_split
        ].reset_index(drop=True)

        if self.samples.empty:
            raise ValueError(f"No processed samples found for split: {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples.iloc[index]
        file_path = Path(row["output_file"])

        with np.load(file_path) as data:
            image = data["image"].copy()
            target = data["target"].copy()
            valid_mask = data["valid_mask"].copy()

        return {
            "image": torch.from_numpy(image).float(),
            "target": torch.from_numpy(target).float().unsqueeze(0),
            "valid_mask": torch.from_numpy(valid_mask).float().unsqueeze(0),
            "sample": row["sample"],
        }
