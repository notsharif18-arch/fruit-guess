"""
FruitVision AI
Predict fruits from images.
"""

from pathlib import Path
import argparse

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import IMAGE_SIZE, MODEL_PATH


def load_class_names():
    train_dir = Path("dataset") / "Training"

    if not train_dir.exists():
        raise FileNotFoundError(
            "Dataset not found. Run setup.bat first."
        )

    classes = sorted(
        [
            folder.name
            for folder in train_dir.iterdir()
            if folder.is_dir()
        ]
    )

    return classes


def preprocess(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image = np.array(image).astype("float32")

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    image = np.expand_dims(image, axis=0)

    return image


def predict(image_path):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Train the model first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    classes = load_class_names()

    image = preprocess(image_path)

    prediction = model.predict(image, verbose=0)[0]

    index = np.argmax(prediction)

    confidence = prediction[index] * 100

    print("=" * 50)
    print("Prediction")
    print("=" * 50)

    print(f"Fruit      : {classes[index]}")
    print(f"Confidence : {confidence:.2f}%")

    print("\nTop 5 Predictions:\n")

    top = np.argsort(prediction)[::-1][:5]

    for i in top:
        print(f"{classes[i]:20s} {prediction[i]*100:.2f}%")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help="Path to image"
    )

    args = parser.parse_args()

    predict(args.image)
