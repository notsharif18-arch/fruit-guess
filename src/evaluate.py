"""
Evaluate trained model
"""

import tensorflow as tf

from src.config import MODEL_PATH
from src.dataloader import load_datasets


def main():

    print("Loading validation data...")

    _, val_ds, classes, _ = load_datasets()

    model = tf.keras.models.load_model(MODEL_PATH)

    loss, accuracy = model.evaluate(
        val_ds,
        verbose=1
    )

    print("=" * 40)
    print("Evaluation Results")
    print("=" * 40)

    print(f"Validation Accuracy : {accuracy*100:.2f}%")
    print(f"Validation Loss     : {loss:.4f}")
    print(f"Classes             : {len(classes)}")

    print("\nDetected Classes:")

    for c in classes:
        print("-", c)


if __name__ == "__main__":
    main()
