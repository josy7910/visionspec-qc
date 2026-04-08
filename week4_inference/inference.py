# week4_inference/inference.py
# WEEK 4 — High-Speed Inference + Live Webcam Demo
# Runs model at >10 FPS on live webcam feed

import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMG_SIZE   = (224, 224)
MODEL_PATH = "models/visionspec_model.h5"

# Colour constants for OpenCV (BGR format)
GREEN  = (57, 153, 39)
RED    = (34, 34, 226)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)


# ──────────────────────────────────────────
# 1. Load Optimized Model
# ──────────────────────────────────────────
def load_model(model_path: str):
    """Load saved model and warm it up for fast inference."""
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Warm-up pass — first inference is always slow due to graph compilation
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("Model loaded and warmed up ✅")
    return model


# ──────────────────────────────────────────
# 2. Preprocess a Single Frame
# ──────────────────────────────────────────
def preprocess_frame(frame):
    """Convert webcam frame to model input tensor."""
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


# ──────────────────────────────────────────
# 3. Draw Result Overlay on Frame
# ──────────────────────────────────────────
def draw_overlay(frame, label: str, confidence: float, fps: float):
    """Draw prediction label, confidence and FPS on the frame."""
    h, w = frame.shape[:2]

    # Background bar at top
    color = GREEN if label == "PASS" else RED
    cv2.rectangle(frame, (0, 0), (w, 60), color, -1)

    # Label text
    text = f"{label}  {confidence*100:.1f}%"
    cv2.putText(frame, text, (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2)

    # FPS counter bottom right
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 130, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    # Border colour
    border_color = GREEN if label == "PASS" else RED
    cv2.rectangle(frame, (0, 0), (w-1, h-1), border_color, 3)

    return frame


# ──────────────────────────────────────────
# 4. Run on Single Image (for testing)
# ──────────────────────────────────────────
def predict_image(image_path: str, model):
    """Predict pass/defect for a single image file."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    start = time.time()
    tensor = preprocess_frame(frame)
    pred   = model.predict(tensor, verbose=0)[0][0]
    latency_ms = (time.time() - start) * 1000

    label      = "PASS" if pred > 0.5 else "DEFECT"
    confidence = pred if pred > 0.5 else 1 - pred

    print(f"\n  Image     : {os.path.basename(image_path)}")
    print(f"  Prediction: {label} ({confidence*100:.1f}% confidence)")
    print(f"  Latency   : {latency_ms:.1f} ms")

    result = draw_overlay(frame.copy(), label, confidence, 1000/latency_ms)
    out_path = f"week4_inference/result_{os.path.basename(image_path)}"
    cv2.imwrite(out_path, result)
    print(f"  Saved     : {out_path}")

    cv2.imshow("VisionSpec QC — Single Image", result)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────
# 5. Live Webcam Demo
# ──────────────────────────────────────────
def run_webcam(model):
    """
    Live webcam inference loop.
    Target: >10 FPS as required by the project spec.
    Press Q to quit.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        print("  Try running predict_on_folder() instead.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nWebcam started — Press Q to quit")
    print("Hold a PCB or any object in front of camera...")

    fps_history = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        # Predict
        tensor = preprocess_frame(frame)
        pred   = model.predict(tensor, verbose=0)[0][0]
        label  = "PASS" if pred > 0.5 else "DEFECT"
        conf   = pred if pred > 0.5 else 1 - pred

        # FPS calculation (rolling average of last 10 frames)
        elapsed = time.time() - start
        fps_history.append(1.0 / (elapsed + 1e-6))
        if len(fps_history) > 10:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # Draw overlay
        frame = draw_overlay(frame, label, conf, avg_fps)

        cv2.imshow("VisionSpec QC — Live Demo (Q to quit)", frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # Q or ESC key
          break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDemo ended. Processed {frame_count} frames.")
    print(f"Average FPS: {sum(fps_history)/len(fps_history):.1f}")


# ──────────────────────────────────────────
# 6. Batch Predict on a Folder
# ──────────────────────────────────────────
def predict_on_folder(folder: str, model):
    """Run inference on all images in a folder — useful if no webcam."""
    results = {"pass": 0, "defect": 0, "errors": 0}
    os.makedirs("week4_inference", exist_ok=True)

    for cls in ["pass", "defect"]:
        cls_folder = os.path.join(folder, cls)
        if not os.path.exists(cls_folder):
            continue
        files = [f for f in os.listdir(cls_folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))][:5]

        for fname in files:
            path = os.path.join(cls_folder, fname)
            frame = cv2.imread(path)
            if frame is None:
                results["errors"] += 1
                continue

            tensor = preprocess_frame(frame)
            pred   = model.predict(tensor, verbose=0)[0][0]
            label  = "PASS" if pred > 0.5 else "DEFECT"
            conf   = pred if pred > 0.5 else 1 - pred

            results[label.lower()] += 1
            print(f"  {fname:30} → {label} ({conf*100:.1f}%)")

    print(f"\nSummary: {results['pass']} PASS, {results['defect']} DEFECT, "
          f"{results['errors']} errors")


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("="*50)
    print("  VISIONSPEC QC — WEEK 4 INFERENCE")
    print("="*50)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        print("  Run week2_model/train.py first!")
        sys.exit(1)

    model = load_model(MODEL_PATH)

    print("\nChoose mode:")
    print("  1 → Live webcam demo")
    print("  2 → Predict on val/ folder")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        run_webcam(model)
    else:
        predict_on_folder("data/val", model)

    print("\nWeek 4 complete! ✅")