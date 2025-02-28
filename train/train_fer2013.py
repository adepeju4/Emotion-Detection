import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Fix the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from data_preprocessing.preprocess import load_fer2013, apply_data_augmentation
from models.cnn_model import create_emotion_model

def train_fer2013_model():
    """Train a model on the FER2013 dataset"""
    # Create directory for saving models if it doesn't exist
    os.makedirs(os.path.dirname(config.FER2013_MODEL_PATH), exist_ok=True)
    
    # Load data
    print("Loading FER2013 dataset...")
    (X_train, y_train), (X_test, y_test) = load_fer2013()
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Apply data augmentation to balance classes
    print("Applying data augmentation...")
    X_train_aug, y_train_aug = apply_data_augmentation(X_train, y_train)
    print(f"Augmented training data shape: {X_train_aug.shape}")
    
    # Create model
    print("Creating model...")
    model = create_emotion_model()
    model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            verbose=1,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            config.FER2013_MODEL_PATH,
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
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
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
    plt.savefig('fer2013_training_history.png')
    plt.close()
    
    return model, history

if __name__ == "__main__":
    train_fer2013_model() 