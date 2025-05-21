import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from preprocessing.fer2013_processor import preprocess_face_for_emotion
from preprocessing.ck_plus_processor import process_ck_plus
from preprocessing.affectnet_processor import process_affectnet

def create_preprocessing_visualization_fer2013(image_path, save_path):
    """Create visualization using FER2013 preprocessing pipeline."""
    # Read original image
    original = cv2.imread(str(image_path))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Get intermediate steps
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (48, 48))
    equalized = cv2.equalizeHist(resized)
    normalized = equalized / 255.0
    
    # Final preprocessed using actual preprocessor
    preprocessed = preprocess_face_for_emotion(original)
    
    create_visualization(original, gray, resized, equalized, preprocessed, 
                        'FER2013 Preprocessing Pipeline', save_path)

def create_preprocessing_visualization_ckplus(image_path, save_path):
    """Create visualization using CK+ preprocessing pipeline."""
    # Read original image
    original = cv2.imread(str(image_path))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Get intermediate steps
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (48, 48))
    
    # CK+ doesn't use histogram equalization
    equalized = resized  # Same as resized for visualization
    
    # Normalize
    normalized = resized.astype('float32') / 255.0
    
    create_visualization(original, gray, resized, equalized, normalized,
                        'CK+ Preprocessing Pipeline', save_path)

def create_preprocessing_visualization_affectnet(image_path, save_path):
    """Create visualization using AffectNet preprocessing pipeline."""
    # Read original image
    original = cv2.imread(str(image_path))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Get intermediate steps
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (48, 48))
    
    # AffectNet doesn't use histogram equalization
    equalized = resized  # Same as resized for visualization
    
    # Normalize
    normalized = resized.astype('float32') / 255.0
    
    create_visualization(original, gray, resized, equalized, normalized,
                        'AffectNet Preprocessing Pipeline', save_path)

def create_visualization(original, gray, resized, equalized, final, title, save_path):
    """Create and save the visualization figure."""
    plt.figure(figsize=(15, 3))
    
    # Original
    plt.subplot(151)
    plt.imshow(original)
    plt.title('1. Original Image')
    plt.axis('off')
    
    # Grayscale
    plt.subplot(152)
    plt.imshow(gray, cmap='gray')
    plt.title('2. Grayscale')
    plt.axis('off')
    
    # Resized
    plt.subplot(153)
    plt.imshow(resized, cmap='gray')
    plt.title('3. Resized (48x48)')
    plt.axis('off')
    
    # Equalized (if applicable)
    plt.subplot(154)
    plt.imshow(equalized, cmap='gray')
    plt.title('4. Hist. Equalized' if np.any(equalized != resized) else '4. No Equalization')
    plt.axis('off')
    
    # Final
    plt.subplot(155)
    if len(final.shape) == 3:
        final = final.squeeze()
    plt.imshow(final, cmap='gray')
    plt.title('5. Final Normalized')
    plt.axis('off')
    
    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create results directory
    results_dir = Path('results/preprocessing_examples')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Process example from each dataset
    datasets = {
        'Fer2013': create_preprocessing_visualization_fer2013,
        'CK+': create_preprocessing_visualization_ckplus,
        'AffectNet': create_preprocessing_visualization_affectnet
    }
    
    emotions = ['happiness', 'anger', 'surprise']  # Example emotions to look for
    
    for dataset_name, preprocessor_func in datasets.items():
        data_dir = Path(f'data/{dataset_name}')
        if data_dir.exists():
            # Try to find an example image
            for emotion in emotions:
                emotion_dir = data_dir / 'train' / emotion
                if not emotion_dir.exists():
                    emotion_dir = data_dir / emotion  # For datasets without train/test split
                
                if emotion_dir.exists():
                    image_files = list(emotion_dir.glob('*.jpg')) + list(emotion_dir.glob('*.png'))
                    if image_files:
                        # Process first image found
                        image_path = image_files[0]
                        save_path = results_dir / f'{dataset_name.lower()}_{emotion}_preprocessing.png'
                        print(f"Processing example from {dataset_name} ({emotion})")
                        preprocessor_func(image_path, save_path)
                        break

if __name__ == "__main__":
    main() 