import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model_multitask import MultiTaskResNet18

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to image")
    ap.add_argument("--ckpt", default="outputs/models/multitask_resnet18_best.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt["img_size"]

    model = MultiTaskResNet18(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        room_logits, q_logit = model(x)
        room_idx = int(room_logits.argmax(dim=1).item())
        room_name = classes[room_idx]
        THRESH = 0.05

        q_prob = float(torch.sigmoid(q_logit).item())
        q_label = "GOOD" if q_prob >= THRESH else "BAD"

    print(f"Room: {room_name}")
    print(f"Quality: {q_label} (prob_good={q_prob:.3f})")

if __name__ == "__main__":
    main()
