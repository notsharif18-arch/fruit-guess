"""
FruitVision AI
Real-time webcam prediction
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.config import MODEL_PATH, IMAGE_SIZE

model = tf.keras.models.load_model(MODEL_PATH)

classes = sorted(
    [
        f.name
        for f in (Path("dataset") / "Training").iterdir()
        if f.is_dir()
    ]
)

cap = cv2.VideoCapture(0)

print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = cv2.resize(rgb, IMAGE_SIZE)

    image = image.astype(np.float32)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)[0]

    idx = np.argmax(prediction)

    confidence = prediction[idx] * 100

    text = f"{classes[idx]} ({confidence:.1f}%)"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("FruitVision AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
