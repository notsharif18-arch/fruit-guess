"""
Dataset Loader
FruitVision AI
"""

from pathlib import Path
import tensorflow as tf

from .config import (
    TRAIN_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    SEED,
)


def load_datasets():

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}"
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_ds
        .cache()
        .shuffle(1000)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds
        .cache()
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, class_names, num_classes
