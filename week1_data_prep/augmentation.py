# week1_data_prep/augmentation.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# ──────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────
IMG_SIZE   = (224, 224)   # ResNet50 expects 224x224
BATCH_SIZE = 32
DATA_DIR   = "data/"      # folder with train/ and val/ subfolders


# ──────────────────────────────────────────
# 2. Data Augmentation Pipeline
# ──────────────────────────────────────────
# Training data → heavily augmented (model sees variety)
train_datagen = ImageDataGenerator(
    rescale=1./255,           # normalize pixels 0-255 → 0-1
    rotation_range=20,        # randomly rotate up to 20 degrees
    zoom_range=0.15,          # randomly zoom in/out
    width_shift_range=0.1,    # shift image left/right
    height_shift_range=0.1,   # shift image up/down
    horizontal_flip=True,     # randomly flip horizontally
    vertical_flip=False,      # don't flip vertically (PCBs have orientation)
    brightness_range=[0.8, 1.2],  # simulate different lighting
    fill_mode="nearest"       # fill empty pixels after rotation
)

# Validation data → NO augmentation (we want real evaluation)
val_datagen = ImageDataGenerator(
    rescale=1./255            # only normalize, nothing else
)


# ──────────────────────────────────────────
# 3. Load Images from Folders
# ──────────────────────────────────────────
def load_data():
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",   # binary = pass or defect (2 classes)
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    print(f"\nClass mapping: {train_generator.class_indices}")
    # Expected: {'defect': 0, 'pass': 1}

    print(f"Training images:   {train_generator.samples}")
    print(f"Validation images: {val_generator.samples}")

    return train_generator, val_generator


# ──────────────────────────────────────────
# 4. Visualize Augmented Images
# ──────────────────────────────────────────
def visualize_augmentation(train_generator, num_images=9):
    """
    Shows original vs augmented versions of PCB images.
    This confirms augmentation looks realistic, not distorted.
    """
    # Get one batch of images
    images, labels = next(train_generator)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Augmented PCB Training Images", fontsize=14)

    label_names = {0: "DEFECT ❌", 1: "PASS ✅"}

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            label = int(labels[i])
            ax.set_title(label_names[label], fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("week1_data_prep/augmented_samples.png", dpi=100)
    plt.show()
    print("Saved: week1_data_prep/augmented_samples.png")


# ──────────────────────────────────────────
# 5. Dataset Statistics
# ──────────────────────────────────────────
def show_dataset_stats(train_generator, val_generator):
    """Shows class distribution to check for imbalance."""
    print("\n" + "="*40)
    print("  DATASET STATISTICS")
    print("="*40)

    for name, gen in [("Train", train_generator), ("Val", val_generator)]:
        total = gen.samples
        classes = gen.class_indices
        print(f"\n  {name} set: {total} images")
        for cls_name, cls_id in classes.items():
            count = np.sum(gen.classes == cls_id)
            pct   = count / total * 100
            print(f"    {cls_name:10} → {count:4} images ({pct:.1f}%)")

    print("="*40)


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("Loading and augmenting data...")
    train_gen, val_gen = load_data()
    show_dataset_stats(train_gen, val_gen)
    visualize_augmentation(train_gen)
    print("\nWeek 1 complete! ✅")
    print("Next: run week2_model/train.py to train the model")