"""
FruitVision AI
Streamlit Web App
"""

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.config import IMAGE_SIZE, MODEL_PATH

st.set_page_config(
    page_title="FruitVision AI",
    page_icon="🍎",
    layout="centered"
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("❌ Model not found. Train the model first using train.py")
        st.stop()

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_classes():

    train_dir = Path("dataset") / "Training"

    if not train_dir.exists():
        st.error("❌ Dataset not found.")
        st.stop()

    return sorted(
        [
            x.name
            for x in train_dir.iterdir()
            if x.is_dir()
        ]
    )


def preprocess(image):

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.array(image).astype(np.float32)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    image = np.expand_dims(image, axis=0)

    return image


model = load_model()

classes = load_classes()

st.title("🍎 FruitVision AI")

st.write(
    "Upload a fruit image and let the AI identify it."
)

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    image = Image.open(uploaded)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        with st.spinner("Analyzing image..."):

            processed = preprocess(image)

            prediction = model.predict(
                processed,
                verbose=0
            )[0]

            idx = np.argmax(prediction)

            confidence = prediction[idx] * 100

            st.success(
                f"Prediction: **{classes[idx]}**"
            )

            st.info(
                f"Confidence: **{confidence:.2f}%**"
            )

            st.subheader("Top 5 Predictions")

            top = np.argsort(prediction)[::-1][:5]

            for i in top:

                st.progress(float(prediction[i]))

                st.write(
                    f"{classes[i]} — {prediction[i]*100:.2f}%"
                )
