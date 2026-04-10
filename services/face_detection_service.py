import cv2
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

face_detector = None

def load_face_detector():
    """Load face detection cascade classifier."""
    global face_detector
    try:
        prototxt_path = "models/deploy.prototxt"
        model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
        
        if not (os.path.exists(prototxt_path) and os.path.exists(model_path)):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_detector = cv2.CascadeClassifier(cascade_path)
            logger.info("Using Haar Cascade face detector")
        else:
            face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            logger.info("Using OpenCV DNN face detector")
    except Exception as e:
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_detector = cv2.CascadeClassifier(cascade_path)
            logger.info(f"Using Haar Cascade face detector (fallback): {e}")
        except Exception as e2:
            logger.error(f"Failed to load face detector: {e2}")
            face_detector = None

    if face_detector is None:
        logger.warning("Face detection not available")
    else:
        logger.info("Face detection loaded successfully")

def get_face_detector():
    """Get the loaded face detector."""
    return face_detector
