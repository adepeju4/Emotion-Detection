import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import argparse
import os

# -----------------------------------------------------------------------------
# Model configuration (mirrors api.py)
# -----------------------------------------------------------------------------
MODEL_CONFIGS = {
    "fer2013": {
        "path": "models/fer2013_model.h5",
        "labels": ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
        "input_shape": (48, 48)
    },
    "ckplus": {
        "path": "models/ckplus_model.h5",
        "labels": ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
        "input_shape": (48, 48)
    },
    "affectnet": {
        "path": "models/affectnet_model.h5",
        "labels": ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"],
        "input_shape": (48, 48)
    },
    "efficientnet": {
        "path": "models/efficientnet_affectnet_final.h5",  # Using AffectNet dataset version
        "labels": ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"],
        "input_shape": (48, 48)
    }
}

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def load_selected_model(model_name: str):
    """Load a Keras model based on a key in MODEL_CONFIGS."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_CONFIGS.keys())}")

    model_path = MODEL_CONFIGS[model_name]["path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Make sure the model is trained and saved at this location."
        )

    print(f"Loading {model_name} model from '{model_path}' …")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.\n")
    return model


def preprocess_frame(frame: np.ndarray, target_size: tuple) -> np.ndarray:
    """Convert BGR frame to grayscale, resize and scale to [0, 1]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    img = resized.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Shape: (1, H, W, 1)
    return img

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def run_webcam_demo(model_name: str, camera_index: int = 0, frame_skip: int = 5):
    model = load_selected_model(model_name)
    labels = MODEL_CONFIGS[model_name]["labels"]
    target_size = MODEL_CONFIGS[model_name]["input_shape"]

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam (index {camera_index})")

    print("Press 'q' to quit.\n")
    frame_count = 0
    latest_predictions = []  # list of tuples: (bbox, label, confidence)
    fps = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam — exiting.")
                break

            # Process every `frame_skip` frames to reduce CPU load
            if frame_count % frame_skip == 0:
                start = time.time()

                # Detect faces in the grayscale version of the frame
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = FACE_CASCADE.detectMultiScale(
                    gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                latest_predictions = []  # reset predictions for this interval
                for (x, y, w, h) in faces:
                    face_roi = frame[y : y + h, x : x + w]
                    img = preprocess_frame(face_roi, target_size)
                    preds = model.predict(img, verbose=0)
                    pred_idx = int(np.argmax(preds))
                    pred_label = labels[pred_idx]
                    confidence = float(np.max(preds))
                    latest_predictions.append(((x, y, w, h), pred_label, confidence))

                fps = 1.0 / (time.time() - start + 1e-6)
            frame_count += 1

            # Draw bounding boxes and labels for the latest predictions
            for (x, y, w, h), label, conf in latest_predictions:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{label}: {conf:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )

            # Show FPS in top-left corner
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )

            cv2.imshow("Emotion Detection (Backend Webcam Test)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam demo terminated.")

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Emotion Detection models using a local webcam.")
    parser.add_argument(
        "--model",
        "-m",
        default="fer2013",
        choices=list(MODEL_CONFIGS.keys()),
        help="Which model to use for prediction.",
    )
    parser.add_argument(
        "--camera-index",
        "-c",
        type=int,
        default=0,
        help="OpenCV camera index to use (default: 0)",
    )
    parser.add_argument(
        "--frame-skip",
        "-s",
        type=int,
        default=5,
        help="Process 1 out of every N frames for performance (default: 5)",
    )

    args = parser.parse_args()
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    run_webcam_demo(args.model, args.camera_index, args.frame_skip) 