import cv2
import numpy as np

def preprocess_face_for_emotion(face_img, target_size=(48, 48)):
    """
    Preprocess a face image for emotion detection
    
    Args:
        face_img: The cropped face image
        target_size: Target size for the model
        
    Returns:
        Preprocessed face ready for the model
    """
    # Convert to grayscale if needed
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_face = face_img
        
    # Resize to target size
    resized_face = cv2.resize(gray_face, target_size)
    
    # Apply histogram equalization to improve contrast
    equalized_face = cv2.equalizeHist(resized_face)
    
    # Normalize pixel values
    normalized_face = equalized_face / 255.0
    
    # Expand dimensions for model
    if len(normalized_face.shape) == 2:
        normalized_face = np.expand_dims(normalized_face, axis=-1)
    
    # Convert to RGB if model expects 3 channels
    if normalized_face.shape[-1] == 1 and target_size[0] == 48:
        normalized_face = np.repeat(normalized_face, 3, axis=-1)
    
    return normalized_face 