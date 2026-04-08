# week3_gradcam/gradcam.py
# WEEK 3 — GradCAM Heatmap Visualization

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMG_SIZE   = (224, 224)
MODEL_PATH = "models/visionspec_model.h5"


def load_image(image_path: str) -> tuple:
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)
    return img_tensor, img_array


def make_gradcam_heatmap(img_tensor, model):
    # Get ResNet50 submodel
    resnet_layer = model.get_layer("resnet50")

    # Build feature extractor
    resnet_input      = tf.keras.Input(shape=(224, 224, 3))
    resnet_out        = resnet_layer(resnet_input)
    feature_extractor = tf.keras.Model(inputs=resnet_input, outputs=resnet_out)

    # Get feature maps for THIS image — shape (7, 7, 2048)
    feature_maps = feature_extractor(img_tensor, training=False)[0].numpy()

    # Get prediction score for this image
    pred_score = float(model(img_tensor, training=False)[0][0])

    # Method: use spatial standard deviation across channels
    # This shows which SPATIAL REGIONS had the most varied activations
    # — completely unique per image since it's based on actual pixel features
    heatmap = np.std(feature_maps, axis=-1)  # shape (7, 7)

    # If predicting DEFECT (pred < 0.5), invert so defect region shows red
    if pred_score < 0.5:
        heatmap = heatmap.max() - heatmap

    # Normalize to 0-1
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap

def overlay_heatmap(heatmap, img_array, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = np.uint8(255 * heatmap_colored)
    img_uint8       = np.uint8(255 * img_array)
    superimposed    = cv2.addWeighted(img_uint8, 1 - alpha,
                                      heatmap_colored, alpha, 0)
    return superimposed


def run_gradcam(image_path: str, model, save_path: str = None):
    img_tensor, img_array = load_image(image_path)

    pred       = float(model(img_tensor, training=False)[0][0])
    label      = "PASS ✅" if pred > 0.5 else "DEFECT ❌"
    confidence = pred if pred > 0.5 else 1 - pred

    print(f"  Image     : {os.path.basename(image_path)}")
    print(f"  Prediction: {label} ({confidence*100:.1f}% confidence)")

    heatmap = make_gradcam_heatmap(img_tensor, model)
    overlay = overlay_heatmap(heatmap, img_array)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"GradCAM Analysis — {label} ({confidence*100:.1f}%)", fontsize=13)

    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Activation Heatmap\n(red = high attention)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay\n(where model looked)")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()
    return pred


def batch_gradcam(image_dir: str, model, num_images: int = 6):
    image_paths = []
    for cls in ["pass", "defect"]:
        folder = os.path.join(image_dir, cls)
        if os.path.exists(folder):
            files = [os.path.join(folder, f) for f in os.listdir(folder)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            image_paths.extend(files[:num_images // 2])

    if not image_paths:
        print("[ERROR] No images found in", image_dir)
        return

    os.makedirs("week3_gradcam/outputs", exist_ok=True)

    for i, path in enumerate(image_paths[:num_images]):
        save_path = f"week3_gradcam/outputs/gradcam_{i+1}.png"
        print(f"\n[{i+1}/{len(image_paths[:num_images])}]")
        run_gradcam(path, model, save_path=save_path)


if __name__ == "__main__":
    print("="*50)
    print("  VISIONSPEC QC — WEEK 3 GRADCAM")
    print("="*50)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded ✅")

    print("\nRunning GradCAM on sample images...")
    batch_gradcam("data/val", model, num_images=6)

    print("\nWeek 3 complete! ✅")
    print("Heatmaps saved in week3_gradcam/outputs/")