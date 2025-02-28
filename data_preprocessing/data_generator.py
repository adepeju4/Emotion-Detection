import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config

def create_data_generator(augmentation=True):
    """Create a data generator with optional augmentation"""
    if augmentation:
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
    else:
        train_datagen = ImageDataGenerator()
        
    # Validation data should not be augmented
    val_datagen = ImageDataGenerator()
    
    return train_datagen, val_datagen

def generate_balanced_batches(X, y, batch_size=config.BATCH_SIZE):
    """Generate balanced batches to handle class imbalance"""
    # Get class indices
    class_indices = [np.where(y[:, i] == 1)[0] for i in range(config.NUM_CLASSES)]
    
    # Find the minimum number of samples per class
    min_samples = min([len(indices) for indices in class_indices if len(indices) > 0])
    
    # Calculate samples per class per batch
    samples_per_class = max(1, batch_size // config.NUM_CLASSES)
    
    while True:
        batch_X = []
        batch_y = []
        
        # For each class, randomly select samples
        for class_idx in range(config.NUM_CLASSES):
            indices = class_indices[class_idx]
            if len(indices) == 0:
                continue
                
            # Randomly select samples for this class
            selected_indices = np.random.choice(indices, size=samples_per_class, replace=True)
            
            batch_X.append(X[selected_indices])
            batch_y.append(y[selected_indices])
        
        # Combine and shuffle
        batch_X = np.concatenate(batch_X, axis=0)
        batch_y = np.concatenate(batch_y, axis=0)
        
        # Shuffle the batch
        indices = np.arange(len(batch_X))
        np.random.shuffle(indices)
        batch_X = batch_X[indices]
        batch_y = batch_y[indices]
        
        yield batch_X, batch_y 