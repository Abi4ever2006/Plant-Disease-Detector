"""
Streamlit web app for Plant Leaf Disease Detection
Run: streamlit run app.py
"""
import json, os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/plant_disease_mobilenet.keras")
CLASSES_PATH = "outputs/class_names.json"
IMG_SIZE = (224, 224)

ADVICE = {
    "healthy": "Leaf looks healthy. Continue regular watering and monitoring.",
    "blight": "Remove infected leaves, avoid overhead watering, and consult a local agri officer about a suitable fungicide.",
    "rust": "Prune affected parts, improve air circulation, and apply a recommended fungicide.",
    "mildew": "Reduce humidity, space plants well, and use a sulphur- or neem-based spray.",
    "spot": "Remove spotted leaves, avoid wetting foliage, and use a copper-based fungicide if it spreads.",
    "scab": "Clear fallen leaves and apply a protectant fungicide early in the season.",
    "virus": "No cure: remove infected plants and control insect vectors such as whiteflies or aphids.",
    "rot": "Improve drainage, remove infected tissue, and avoid injuring fruit or roots.",
    "mold": "Increase ventilation, reduce humidity, and remove affected leaves.",
    "mite": "Spray with water, neem oil, or a miticide; keep plants well watered.",
}

def advice_for(label):
    low = label.lower()
    for key, tip in ADVICE.items():
        if key in low:
            return tip
    return "Consult a local agricultural extension officer for confirmation and treatment."

@st.cache_resource
def load():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH) as f:
        classes = json.load(f)
    return model, classes

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Leaf Disease Detector")
st.write("Upload a clear photo of a single leaf to get a predicted diagnosis.")

if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH)):
    st.error("Trained model not found. Run train.py first.")
    st.stop()

model, classes = load()
file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)
    arr = np.expand_dims(np.array(img.resize(IMG_SIZE), dtype="float32"), 0)
    probs = model.predict(arr)[0]
    top = probs.argsort()[::-1][:3]
    label = classes[top[0]].replace("___", " - ").replace("_", " ")
    st.subheader(f"Prediction: {label}")
    st.progress(float(probs[top[0]]))
    st.write(f"Confidence: **{probs[top[0]] * 100:.2f}%**")
    st.info(advice_for(classes[top[0]]))
    with st.expander("Top 3 predictions"):
        for i in top:
            st.write(f"{classes[i].replace('___', ' - ').replace('_', ' ')}: {probs[i] * 100:.2f}%")
    if probs[top[0]] < 0.6:
        st.warning("Low confidence. Try a clearer, well-lit, close-up photo of one leaf.")
