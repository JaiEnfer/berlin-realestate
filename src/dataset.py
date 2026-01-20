import os
from pathlib import Path
from typing import Tuple, List

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(root: Path, classes: List[str]) -> List[Tuple[str, int]]:
    items = []
    for idx, cls in enumerate(classes):
        cls_dir = root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                items.append((str(p), idx))
    return items


class RoomsDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y, path
