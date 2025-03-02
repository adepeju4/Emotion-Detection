import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Fix the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from data_preprocessing.preprocess import load_affectnet

def create_simple_cnn_model(input_shape=(48, 48, 3), num_classes=8):
    """Create a simple CNN model for emotion recognition"""
    model = Sequential([
        # First convolutional block
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

def compute_class_weights(y_train):
    """Compute class weights to handle class imbalance"""
    # Convert one-hot encoded labels to class indices
    y_indices = np.argmax(y_train, axis=1)
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_indices),
        y=y_indices
    )
    
    # Convert to dictionary
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    return class_weight_dict

def train_affectnet_simple():
    """Train a simple CNN model on the AffectNet dataset"""
    print("Training simple CNN model on AffectNet dataset...")
    
    # Set emotion classes to AffectNet specific emotions
    config.EMOTION_CLASSES = config.AFFECTNET_EMOTIONS
    config.NUM_CLASSES = len(config.EMOTION_CLASSES)
    
    print(f"Using emotion classes: {config.EMOTION_CLASSES}")
    
    # Load data
    (X_train, y_train), (X_val, y_val) = load_affectnet()
    
    # Check if data was loaded successfully
    if len(X_train) == 0 or len(X_val) == 0:
        print("Error: Failed to load AffectNet dataset. Skipping model training.")
        return None, None
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Ensure pixel values are in [0, 1]
    if np.max(X_train) > 1.0:
        X_train = X_train / 255.0
        X_val = X_val / 255.0
    
    # Convert grayscale to RGB if needed
    if X_train.shape[-1] == 1:
        X_train = np.repeat(X_train, 3, axis=-1)
        X_val = np.repeat(X_val, 3, axis=-1)
        print(f"Converted grayscale to RGB. New shape: {X_train.shape}")
    
    # Create model
    model = create_simple_cnn_model(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, X_train.shape[-1]),
        num_classes=config.NUM_CLASSES
    )
    
    # Create directory for saving models if it doesn't exist
    model_path = os.path.join(config.MODELS_SAVE_DIR, 'affectnet_simple_model.h5')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Compute class weights to handle class imbalance
    class_weights = compute_class_weights(y_train)
    print("Class weights:", class_weights)
    
    print("Model summary:")
    model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights
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
    plt.savefig('affectnet_simple_training_history.png')
    plt.close()
    
    return model, history

if __name__ == "__main__":
    train_affectnet_simple() 