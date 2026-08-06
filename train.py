"""
Train the FruitVision AI model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import MODEL_PATH
from src.dataloader import load_datasets
from src.model import build_model, unfreeze_model


def plot_history(history, filename):
    Path("assets").mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))

    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(Path("assets") / filename)
    plt.close()


def main():

    print("=" * 50)
    print(" FruitVision AI")
    print("=" * 50)

    train_ds, val_ds, class_names, num_classes = load_datasets()

    print(f"Detected {num_classes} fruit classes")
    print(class_names)

    model = build_model(num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    print("\nTraining head...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
    )

    plot_history(history, "training_accuracy.png")

    print("\nFine-tuning model...")

    model = unfreeze_model(model)

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=callbacks,
    )

    plot_history(history_ft, "finetune_accuracy.png")

    print("\nTraining Complete!")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
