from pathlib import Path
from typing import List, Tuple
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class MultiTaskRoomsDataset(Dataset):
    """
    Returns:
      x: image tensor
      y_room: int (0..4)
      y_quality: float (0 or 1)
      path: str
    """
    def __init__(self, csv_path: str, classes: List[str], transform=None, indices=None):
        self.df = pd.read_csv(csv_path)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

        # Clean + map room -> index
        self.df["room_idx"] = self.df["room"].map(self.class_to_idx)

        # Remove anything weird (just in case)
        self.df = self.df.dropna(subset=["path", "room_idx", "quality_label"]).reset_index(drop=True)

        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = row["path"]
        room_idx = int(row["room_idx"])
        quality = float(row["quality_label"])  # binary 0/1

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, room_idx, quality, path
