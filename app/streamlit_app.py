

import os
import sys

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from dataset import CIFAR10_MEAN, CIFAR10_STD, CLASS_NAMES  
from model import build_model  

st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️")


@st.cache_resource
def load_model():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = build_model()
    checkpoint_path = os.path.join(PROJECT_ROOT,"src","checkpoints", "best_model.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image: Image.Image):
    """Resize to 32x32 and normalize, matching training-time preprocessing."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return transform(image).unsqueeze(0)  # add batch dimension


def main():
    st.title("🖼️ CIFAR-10 Image Classifier")
    st.write(
        "Upload an image and the model will classify it into one of 10 categories: "
        + ", ".join(CLASS_NAMES)
    )

    model, device = load_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        input_tensor = preprocess_image(image).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze().cpu()

        top_prob, top_idx = torch.max(probabilities, dim=0)
        st.subheader(f"Prediction: **{CLASS_NAMES[top_idx]}** ({top_prob.item() * 100:.1f}% confidence)")

        st.write("All class probabilities:")
        prob_dict = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        st.bar_chart(sorted_probs)

        st.caption(
            "Note: this model is trained on CIFAR-10 (32x32 low-resolution images), "
            "so predictions on high-resolution real-world photos may be less reliable "
            "than on images resembling the original dataset's style and framing."
        )


if __name__ == "__main__":
    main()