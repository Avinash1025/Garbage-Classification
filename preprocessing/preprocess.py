import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.config import RAW_DATA_DIR, IMAGE_SIZE, BATCH_SIZE, SEED

def preprocess_data():
    print("[INFO] Starting preprocessing...")

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Train generator
    train_gen = datagen.flow_from_directory(
        RAW_DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        subset="training",
        seed=SEED,
        shuffle=True,
        class_mode="categorical"
    )

    # Validation generator
    val_gen = datagen.flow_from_directory(
        RAW_DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        subset="validation",
        seed=SEED,
        shuffle=True,
        class_mode="categorical"
    )

    print(f"[INFO] Found {train_gen.samples} training and {val_gen.samples} validation samples")
    print(f"[INFO] Class mapping: {train_gen.class_indices}")

    return train_gen, val_gen

if __name__ == "__main__":
    preprocess_data()
