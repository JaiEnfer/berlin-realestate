# app.py
# Streamlit demo: Room Type (5-class) + Photo Quality (binary) using your multi-task CNN checkpoint.
#
# Run:
#   pip install streamlit
#   streamlit run app.py
#
# Notes:
# - Uses the TensorBoard-trained checkpoint by default:
#     outputs/models/multitask_resnet18_best_tb.pt
# - Default quality threshold is 0.05 (from your best-F1 threshold search)

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Import your model (expects src/model_multitask.py)
from src.model_multitask import MultiTaskResNet18


st.set_page_config(page_title="Berlin Real Estate: Room + Quality", layout="centered")
st.title("🏠 Berlin Real Estate — Room Type + Photo Quality (Multi-task CNN)")
st.caption("Upload an interior photo. The model predicts room type and a proxy photo-quality label.")


# -------------------------
# Config (editable in UI)
# -------------------------
DEFAULT_CKPT = "outputs/models/multitask_resnet18_best_tb.pt"
DEFAULT_THRESHOLD = 0.05

with st.sidebar:
    st.header("Settings")
    ckpt_path = st.text_input("Checkpoint path", value=DEFAULT_CKPT)
    threshold = st.slider("Quality threshold", 0.0, 1.0, float(DEFAULT_THRESHOLD), 0.01)
    show_probs = st.checkbox("Show room probabilities", value=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"**Device:** {device}")


@st.cache_resource
def load_model_and_transforms(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    classes = ckpt.get("classes", ["Bathroom", "Bedroom", "Dinning", "Kitchen", "Livingroom"])
    img_size = int(ckpt.get("img_size", 224))

    model = MultiTaskResNet18(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, classes, tfm


# -------------------------
# Load model
# -------------------------
model = None
classes = None
tfm = None

try:
    model, classes, tfm = load_model_and_transforms(ckpt_path)
    st.sidebar.success("Model loaded ✅")
except Exception as e:
    st.sidebar.error("Failed to load model/checkpoint.")
    st.sidebar.exception(e)


# -------------------------
# UI: Upload + Predict
# -------------------------
uploaded = st.file_uploader("Upload an interior photo", type=["jpg", "jpeg", "png", "webp"])

col1, col2 = st.columns(2)

if uploaded and model is not None:
    img = Image.open(uploaded).convert("RGB")

    with col1:
        st.image(img, caption="Uploaded image", use_container_width=True)

    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        room_logits, q_logit = model(x)

        room_probs = torch.softmax(room_logits, dim=1).cpu().numpy().flatten()
        room_idx = int(room_probs.argmax())
        room_name = classes[room_idx]

        q_prob_good = float(torch.sigmoid(q_logit).item())
        quality_label = "GOOD" if q_prob_good >= threshold else "BAD"

    with col2:
        st.subheader("Prediction")
        st.write(f"**Room type:** {room_name}")
        st.write(f"**Photo quality:** {quality_label}")
        st.write(f"**P(GOOD):** {q_prob_good:.3f}")
        st.write(f"**Threshold:** {threshold:.2f}")

        st.divider()

        # Quick interpretation text (optional)
        if quality_label == "GOOD":
            st.success("Looks like a reasonably clear / well-lit photo (proxy metric).")
        else:
            st.warning("May be dark, blurry, or low-resolution (proxy metric).")

    if show_probs:
        st.subheader("Room class probabilities")
        prob_dict = {classes[i]: float(room_probs[i]) for i in range(len(classes))}
        st.bar_chart(prob_dict)

else:
    st.info("Upload an image to get predictions.")

st.markdown("---")
st.caption(
    "Quality labels are proxy labels derived from blur/brightness/resolution heuristics. "
    "For a production version, label a human-rated subset and calibrate the threshold."
)
