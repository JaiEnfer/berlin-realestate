from pathlib import Path
import shutil
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_multitask import MultiTaskRoomsDataset
from model_multitask import MultiTaskResNet18

CLASSES = ["Bathroom", "Bedroom", "Dinning", "Kitchen", "Livingroom"]
CSV_PATH = "data/processed/quality.csv"
CKPT_PATH = "outputs/models/multitask_resnet18_best_tb.pt"

OUT_DIR = Path("outputs/failures_room")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT_PATH, map_location=device)

    model = MultiTaskResNet18(num_classes=len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = MultiTaskRoomsDataset(CSV_PATH, CLASSES, transform=tfm)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    mistakes = []
    with torch.no_grad():
        for x, y_room, y_q, paths in tqdm(loader, desc="Scanning"):
            x = x.to(device)
            room_logits, _ = model(x)
            pred = room_logits.argmax(dim=1).cpu().numpy().tolist()

            for p, yt, yp in zip(paths, y_room, pred):
                if int(yt) != int(yp):
                    mistakes.append((p, int(yt), int(yp)))

    # Save first 30 mistakes (or all if fewer)
    for i, (p, yt, yp) in enumerate(mistakes[:30], start=1):
        src = Path(p)
        dst = OUT_DIR / f"{i:02d}_true-{CLASSES[yt]}_pred-{CLASSES[yp]}{src.suffix.lower()}"
        try:
            shutil.copy(src, dst)
        except Exception:
            pass

    print(f"Total mistakes found: {len(mistakes)}")
    print(f"Saved up to 30 examples into: {OUT_DIR}")

if __name__ == "__main__":
    main()
