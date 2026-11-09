
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model_daun_mint_terbaik.keras"
LABEL_PATH = BASE_DIR / "label_kelas.json"

st.set_page_config(
    page_title="CNN Daun Mint",
    page_icon="🌿",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )

    with open(LABEL_PATH, "r") as f:
        labels = json.load(f)

    return model, labels

st.title("🌿 Klasifikasi Daun Mint")
st.write("Deteksi kondisi daun mint menggunakan CNN.")

try:
    model, labels = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

class_names = labels["class_names"]
positive_class = labels["positive_class"]
img_size = tuple(labels["image_size"])
threshold = labels["threshold"]

uploaded_file = st.file_uploader(
    "Upload gambar daun mint",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Gambar yang diunggah",
             use_container_width=True)

    if st.button("Prediksi Daun Mint"):
        resized = image.resize(img_size)

        img_array = np.array(
            resized, dtype=np.float32
        ) / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        prediction = np.asarray(
            model.predict(img_array, verbose=0)
        ).reshape(-1)

        if prediction.size == 1:
            probability = float(prediction[0])

            predicted_class = (
                positive_class
                if probability >= threshold
                else next(c for c in class_names
                          if c != positive_class)
            )

            confidence = max(probability, 1 - probability)

        elif prediction.size == len(class_names):
            index = int(np.argmax(prediction))
            predicted_class = class_names[index]
            confidence = float(prediction[index])

        else:
            st.error("Format output model tidak dikenali.")
            st.stop()

        st.success(f"Hasil prediksi: {predicted_class}")

        st.metric(
            "Skor prediksi model",
            f"{confidence * 100:.2f}%"
        )

        st.caption(
            "Skor model bukan jaminan kebenaran prediksi."
        )
