import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from preprocessing.fer2013_processor import process_fer2013
from preprocessing.ck_plus_processor import process_ck_plus
from preprocessing.affectnet_processor import process_affectnet

def visualize_dataset_preprocessing(dataset_name, processor_func, sample_image_path):
    plt.figure(figsize=(15, 3))
    
    original = cv2.imread(str(sample_image_path))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    plt.subplot(141)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    if dataset_name == 'FER2013':
        processed_data = processor_func('data/Fer2013', ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'])
        processed_image = processed_data[0][0] 
    elif dataset_name == 'CK+':
        processed_data = processor_func('data/CK+', ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'])
        processed_image = processed_data[0][0] 
    elif dataset_name == 'AffectNet':
        processed_data = processor_func('data/Affectnet', ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'])
        processed_image = processed_data[0][0]
    else:  
        image = cv2.imread(str(sample_image_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (48, 48))
        image = np.array([image])
        image = image.astype('float32') / 255.0
        processed_image = image.reshape(1, 48, 48, 1)[0]
    
    plt.subplot(142)
    plt.imshow(processed_image[:,:,0], cmap='gray')  
    plt.title('Processed (48x48, Grayscale, Normalized)')
    plt.axis('off')
    
    plt.suptitle(f'{dataset_name} Preprocessing Pipeline', y=1.05)
    plt.tight_layout()
    
    results_dir = Path('results/preprocessing_pipeline')
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / f'{dataset_name.lower()}_preprocessing.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    processors = {
        'FER2013': process_fer2013,
        'CK+': process_ck_plus,
        'AffectNet': process_affectnet,
        'Real_World': None  
    }
    
    print("Visualizing preprocessing pipelines...")
    
    for dataset_name, processor_func in processors.items():
        if dataset_name == 'Real_World':
            data_dir = Path('data/real_world_samples')
            if not data_dir.exists():
                print("\nNo real-world samples found. Please add images to data/real_world_samples/")
                continue
        else:
            data_dir = Path(f'data/{dataset_name}')
        
        if data_dir.exists():
            image_files = list(data_dir.rglob('*.jpg')) + list(data_dir.rglob('*.png'))
            if image_files:
                print(f"\nProcessing sample from {dataset_name}...")
                visualize_dataset_preprocessing(dataset_name, processor_func, image_files[0])
                print(f"Saved visualization for {dataset_name}")

if __name__ == "__main__":
    main() 