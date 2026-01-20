from pathlib import Path
import csv
import cv2
import numpy as np

CLASSES = ["Bathroom", "Bedroom", "Dinning", "Kitchen", "Livingroom"]
DATA_DIR = Path("data/raw/House_Room_Dataset")
OUT_CSV = Path("data/processed/quality.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def score_image(img_bgr):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Sharpness proxy: higher is sharper
    sharp = variance_of_laplacian(gray)
    sharp_norm = np.clip((sharp - 20) / (300 - 20), 0, 1)

    # Brightness proxy: penalize too dark or too bright
    mean_brightness = gray.mean() / 255.0
    bright_penalty = abs(mean_brightness - 0.5) * 2  # 0 is best, 1 is worst
    bright_norm = 1 - np.clip(bright_penalty, 0, 1)

    # Resolution proxy
    res = h * w
    res_norm = np.clip((res - 100_000) / (1_000_000 - 100_000), 0, 1)

    # Weighted score
    score = 0.45 * sharp_norm + 0.35 * bright_norm + 0.20 * res_norm
    return float(score)


def main():
    rows = []
    for cls in CLASSES:
        for p in (DATA_DIR / cls).rglob("*"):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            s = score_image(img)
            rows.append((str(p), cls, s))

    rows.sort(key=lambda x: x[0])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "room", "quality_score", "quality_label"])
        for path, room, score in rows:
            # Convert score to label (good/bad) with threshold
            label = 1 if score >= 0.55 else 0
            w.writerow([path, room, f"{score:.4f}", label])

    print(f"✅ Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
