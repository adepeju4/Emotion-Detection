from __future__ import annotations

import numpy as np
import cv2
import io
import logging
from typing import Dict, List, TYPE_CHECKING
from fastapi import HTTPException

from services.model_service import get_model, get_model_config, predict as model_predict
from services.face_detection_service import get_face_detector

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# Lazy import — PIL takes ~4.5s to load
_Image = None
def _get_pil():
    global _Image
    if _Image is None:
        from PIL import Image as _PILImage
        _Image = _PILImage
    return _Image

def determine_face_orientation(face_image: Image.Image) -> Image.Image:
    """Determine correct face orientation (simplified without MediaPipe)."""
    try:
        aspect_ratio = face_image.width / face_image.height
        if aspect_ratio > 1.5:
            return face_image.rotate(-90, expand=True)
        elif aspect_ratio < 0.7:
            return face_image.rotate(90, expand=True)
        else:
            return face_image
    except Exception as e:
        return face_image

def preprocess_image(image: Image.Image, target_size: tuple) -> np.ndarray:
    """Preprocess the image for model input."""
    try:
        if image.mode != "L":
            image = image.convert("L")

        image = image.resize(target_size)

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Failed to preprocess image")

def validate_image(contents: bytes) -> Image.Image:
    """Validate and open image from bytes, fixing EXIF orientation."""
    Image = _get_pil()
    try:
        image = Image.open(io.BytesIO(contents))

        try:
            if hasattr(image, '_getexif') and image._getexif() is not None:
                exif = image._getexif()
                orientation = exif.get(274)
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        except Exception:
            pass

        return image
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        raise HTTPException(status_code=400, detail="Failed to process image")

def detect_and_analyze_faces(image: Image.Image, model_name: str) -> List[Dict]:
    """Detect faces in image and analyze emotions for each face using OpenCV."""
    Image = _get_pil()
    try:
        face_detector = get_face_detector()
        if face_detector is None:
            logger.error("Face detector not available")
            return []

        model = get_model(model_name)
        config = get_model_config(model_name)

        if model is None or config is None:
            raise HTTPException(status_code=503, detail=f"Model '{model_name}' is not available")

        img_rgb = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_height, img_width = img_cv.shape[:2]

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        if isinstance(face_detector, cv2.dnn.Net):
            blob = cv2.dnn.blobFromImage(img_cv, 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_detector.setInput(blob)
            detections = face_detector.forward()
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.5:
                    continue
                x1 = int(detections[0, 0, i, 3] * img_width)
                y1 = int(detections[0, 0, i, 4] * img_height)
                x2 = int(detections[0, 0, i, 5] * img_width)
                y2 = int(detections[0, 0, i, 6] * img_height)
                faces.append((x1, y1, x2 - x1, y2 - y1, float(confidence)))
        elif isinstance(face_detector, cv2.CascadeClassifier):
            detections = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            faces = [(x, y, w, h, 1.0) for (x, y, w, h) in detections]
        else:
            faces = []

        if len(faces) == 0:
            return []

        target_size = config["input_shape"]
        face_results = []
        face_id = 1

        for (x, y, w, h, confidence) in faces:
            if confidence < 0.5:
                logger.info(f"Skipping low confidence detection: {confidence:.3f}")
                continue

            x = int(max(0, int(x)))
            y = int(max(0, int(y)))
            w = int(min(int(w), img_width - x))
            h = int(min(int(h), img_height - y))

            if w < 20 or h < 20 or w > img_width * 0.9 or h > img_height * 0.9:
                continue

            aspect_ratio = w / h
            if aspect_ratio < 0.6 or aspect_ratio > 1.8:
                continue

            face_roi = img_rgb[y:y+h, x:x+w]
            face_pil = Image.fromarray(face_roi)
            face_pil_oriented = face_pil

            face_array = preprocess_image(face_pil_oriented, target_size)

            preds = model_predict(model_name, face_array)
            pred_class = int(np.argmax(preds))
            pred_label = config["labels"][pred_class]
            confidence = float(np.max(preds))

            all_predictions = {
                label: float(conf)
                for label, conf in zip(config["labels"], preds[0])
            }
            sorted_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))

            face_results.append({
                "face_id": int(face_id),
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "label": str(pred_label),
                "confidence": float(confidence),
                "all_predictions": {k: float(v) for k, v in sorted_predictions.items()},
            })
            face_id += 1

        return face_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in face detection and analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to detect faces")
