import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Fix the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from data_preprocessing.preprocess import load_affectnet

def create_transfer_learning_model(base_model_name='efficientnet', num_classes=8):
    """Create a transfer learning model for emotion recognition
    
    Args:
        base_model_name: 'mobilenetv2' or 'efficientnet'
        num_classes: Number of emotion classes
    """
    input_shape = (config.IMG_SIZE, config.IMG_SIZE, 3)
    
    # Select base model
    if base_model_name.lower() == 'mobilenetv2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name.lower() == 'efficientnet':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head with stronger regularization
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # First dense block with stronger regularization
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Second dense block
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Output layer
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model

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

def preprocess_images(images):
    """Apply additional preprocessing to images"""
    # Ensure pixel values are in [0, 1]
    if np.max(images) > 1.0:
        images = images / 255.0
    
    # Apply normalization for pretrained models
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    
    return images

def train_affectnet_improved(base_model_name='efficientnet', fine_tune=True):
    """Train an improved model on the AffectNet dataset using transfer learning"""
    print(f"Training improved AffectNet model using {base_model_name}...")
    
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
    
    # Convert grayscale to RGB if needed
    if X_train.shape[-1] == 1:
        X_train = np.repeat(X_train, 3, axis=-1)
        X_val = np.repeat(X_val, 3, axis=-1)
        print(f"Converted grayscale to RGB. New shape: {X_train.shape}")
    
    # Apply additional preprocessing
    X_train = preprocess_images(X_train)
    X_val = preprocess_images(X_val)
    
    # Create model
    model, base_model = create_transfer_learning_model(
        base_model_name=base_model_name,
        num_classes=config.NUM_CLASSES
    )
    
    # Create directory for saving models if it doesn't exist
    model_path = os.path.join(config.MODELS_SAVE_DIR, f'affectnet_{base_model_name}_model.h5')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Compute class weights to handle class imbalance
    class_weights = compute_class_weights(y_train)
    print("Class weights:", class_weights)
    
    # Compile model with categorical crossentropy loss
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Lower initial learning rate
        loss=CategoricalCrossentropy(label_smoothing=0.2),  # Increased label smoothing
        metrics=['accuracy']
    )
    
    print("Model summary:")
    model.summary()
    
    # Define callbacks with more patience
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # More patience
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,  # More patience
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
    
    # Train the model (first phase - train only the top layers)
    print("Phase 1: Training only the top layers...")
    history1 = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=25,  # More epochs
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Fine-tuning phase
    if fine_tune:
        print("Phase 2: Fine-tuning the model...")
        # Unfreeze some layers of the base model
        if base_model_name.lower() == 'mobilenetv2':
            # Unfreeze the last 23 layers (last 3 blocks of MobileNetV2)
            for layer in base_model.layers[-23:]:
                layer.trainable = True
        elif base_model_name.lower() == 'efficientnet':
            # Unfreeze the last 30 layers
            for layer in base_model.layers[-30:]:
                layer.trainable = True
        
        # Recompile with a lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.00005),  # Even lower learning rate for fine-tuning
            loss=CategoricalCrossentropy(label_smoothing=0.2),
            metrics=['accuracy']
        )
        
        # Continue training
        history2 = model.fit(
            X_train, y_train,
            batch_size=16,  # Smaller batch size for fine-tuning
            epochs=40,  # More epochs for fine-tuning
            initial_epoch=len(history1.history['accuracy']),
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        # Combine histories
        history = {}
        for k in history1.history.keys():
            history[k] = history1.history[k] + history2.history[k]
    else:
        history = history1.history
    
    # Evaluate the model
    print("Evaluating model...")
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'affectnet_{base_model_name}_training_history.png')
    plt.close()
    
    return model, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train improved AffectNet model')
    parser.add_argument('--model', type=str, default='efficientnet', 
                        choices=['mobilenetv2', 'efficientnet'],
                        help='Base model architecture')
    parser.add_argument('--no-fine-tune', action='store_true',
                        help='Skip fine-tuning phase')
    
    args = parser.parse_args()
    
    train_affectnet_improved(
        base_model_name=args.model,
        fine_tune=not args.no_fine_tune
    ) 