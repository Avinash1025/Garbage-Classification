import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
from models.efficientnet_model import EfficientLite

st.title("🗑️ Garbage Classification using EfficientNet")

# ---------- Load Checkpoint ----------
@st.cache_resource
def load_model():
    checkpoint_list = glob.glob("checkpoints/*.ckpt")
    if not checkpoint_list:
        st.error("❌ Model checkpoint not found in 'checkpoints/' directory.")
        return None

    checkpoint_path = checkpoint_list[0]

    # ✅ FIXED: Match num_class to what model was trained on (6)
    model = EfficientLite.load_from_checkpoint(checkpoint_path, lr=3e-5, num_class=6)
    model.eval()
    return model

model = load_model()

# ✅ FIXED: Correct class list for 6 classes
class_names = [
    "Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"
]

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

uploaded_file = st.file_uploader("Upload a garbage image...", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)
        predicted_label = class_names[top_class.item()]
        confidence = top_prob.item() * 100

    st.success(f"✅ Predicted: **{predicted_label}** with {confidence:.2f}% confidence")
