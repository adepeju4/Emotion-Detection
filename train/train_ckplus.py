import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Fix the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from data_preprocessing.preprocess import load_ckplus, apply_data_augmentation
from models.cnn_model import create_lightweight_model, create_emotion_model

def train_ckplus_model():
    """Train a model on the CK+ dataset"""
    print("Training CK+ model...")
    
    # Set emotion classes to CK+ specific emotions
    config.EMOTION_CLASSES = config.CKPLUS_EMOTIONS
    config.NUM_CLASSES = len(config.EMOTION_CLASSES)
    
    print(f"Using emotion classes: {config.EMOTION_CLASSES}")
    
    # Load data
    (X_train, y_train), (X_val, y_val) = load_ckplus()
    
    # Create model
    model = create_emotion_model(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS),
        num_classes=config.NUM_CLASSES
    )
    
    # Create directory for saving models if it doesn't exist
    os.makedirs(os.path.dirname(config.CKPLUS_MODEL_PATH), exist_ok=True)
    
    # Apply data augmentation to balance classes
    print("Applying data augmentation...")
    X_train_aug, y_train_aug = apply_data_augmentation(X_train, y_train, augmentation_factor=5)
    print(f"Augmented training data shape: {X_train_aug.shape}")
    
    # Create model - using lightweight model since CK+ is smaller
    print("Creating model...")
    model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=7,
            verbose=1,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            config.CKPLUS_MODEL_PATH,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Train the model
    print("Training model...")
    history = model.fit(
        X_train_aug, y_train_aug,
        batch_size=config.BATCH_SIZE // 2,  # Smaller batch size for smaller dataset
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    # Evaluate the model
    print("Evaluating model...")
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('ckplus_training_history.png')
    plt.close()
    
    return model, history

if __name__ == "__main__":
    train_ckplus_model() 