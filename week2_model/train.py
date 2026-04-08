# week2_model/train.py
# WEEK 2 — Transfer Learning Model Training
# Trains ResNet50-based binary classifier for pass/defect detection

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (EarlyStopping,
                                         ModelCheckpoint,
                                         ReduceLROnPlateau)
import matplotlib.pyplot as plt
import numpy as np
import os, json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from week1_data_prep.augmentation import load_data

# ──────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 20
MODEL_PATH = "models/visionspec_model.h5"
os.makedirs("models", exist_ok=True)


# ──────────────────────────────────────────
# 2. Build the Model (Transfer Learning)
# ──────────────────────────────────────────
def build_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze ALL base layers — don't touch pre-trained weights at all
    base_model.trainable = False

    print(f"Base model layers : {len(base_model.layers)}")
    print(f"Trainable layers  : 0 (fully frozen)")

    inputs  = tf.keras.Input(shape=(224, 224, 3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs=x, name="VisionSpec_ResNet50")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    return model


# ──────────────────────────────────────────
# 3. Training Callbacks
# ──────────────────────────────────────────
def get_callbacks():
    return [
        EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor="val_loss",
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        ReduceLROnPlateau(
            patience=2,
            factor=0.5,
            monitor="val_loss",
            verbose=1,
            min_lr=1e-7
        )
    ]


# ──────────────────────────────────────────
# 4. Train the Model
# ──────────────────────────────────────────
def train_model(model, train_gen, val_gen):
    print("\nStarting training...")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print("-" * 50)

    # Calculate class weights to handle imbalance
    total    = train_gen.samples
    n_defect = int(np.sum(train_gen.classes == 0))
    n_pass   = int(np.sum(train_gen.classes == 1))
    w_defect = total / (2 * n_defect)
    w_pass   = total / (2 * n_pass)
    class_weights = {0: w_defect, 1: w_pass}
    print(f"Class weights: defect={w_defect:.2f}, pass={w_pass:.2f}")

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks(),
        class_weight=class_weights,
        verbose=1
    )
    return history


# ──────────────────────────────────────────
# 5. Plot Learning Curves
# ──────────────────────────────────────────
def plot_learning_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VisionSpec QC — Learning Curves", fontsize=14)

    axes[0].plot(history.history["loss"],     label="Train Loss",     color="#534AB7")
    axes[0].plot(history.history["val_loss"], label="Val Loss",       color="#D85A30", linestyle="--")
    axes[0].set_title("Loss (lower = better)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"],     label="Train Accuracy", color="#534AB7")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy",   color="#D85A30", linestyle="--")
    axes[1].set_title("Accuracy (higher = better)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("week2_model/learning_curves.png", dpi=100)
    plt.show()
    print("Saved: week2_model/learning_curves.png")


# ──────────────────────────────────────────
# 6. Evaluate the Model
# ──────────────────────────────────────────
def evaluate_model(model, val_gen):
    print("\n" + "="*50)
    print("  MODEL EVALUATION")
    print("="*50)

    results = model.evaluate(val_gen, verbose=0)
    metrics = dict(zip(model.metrics_names, results))

    print(f"  Loss      : {metrics['loss']:.4f}")
    print(f"  Accuracy  : {results[1]*100:.2f}%")
    print(f"  Precision : {results[2]*100:.2f}%")
    print(f"  Recall    : {results[3]*100:.2f}%")

    # F1 Score
    p  = results[2]
    r  = results[3]
    f1 = 2 * (p * r) / (p + r + 1e-7)
    print(f"  F1 Score  : {f1*100:.2f}%")
    print("="*50)

    # Save metrics
    output = {
        "loss":      float(results[0]),
        "accuracy":  float(results[1]),
        "precision": float(results[2]),
        "recall":    float(results[3]),
        "f1":        float(f1)
    }
    with open("week2_model/metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    print("  Metrics saved to week2_model/metrics.json")
    return output


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("="*50)
    print("  VISIONSPEC QC — WEEK 2 MODEL TRAINING")
    print("="*50)

    # Load data from Week 1
    train_gen, val_gen = load_data()

    # Build model
    model = build_model()
    model.summary()

    # Train
    history = train_model(model, train_gen, val_gen)

    # Evaluate
    evaluate_model(model, val_gen)

    # Plot
    plot_learning_curves(history)

    print("\nWeek 2 complete! ✅")
    print(f"Model saved at: {MODEL_PATH}")
    print("Next: run week3_gradcam/gradcam.py for heatmap visualization")
    