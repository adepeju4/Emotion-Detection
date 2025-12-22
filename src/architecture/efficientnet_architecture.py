import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from utils.utils import determine_num_classes
import numpy as np

class ResizeLayer(layers.Layer):
    def __init__(self, target_size, **kwargs):
        super().__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size)

def cosine_decay_with_warmup(epoch, total_epochs, warmup_epochs=5, learning_rate_max=0.001):
    """Custom learning rate schedule with warmup and cosine decay"""
    if epoch < warmup_epochs:
        # Linear warmup
        return learning_rate_max * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return learning_rate_max * 0.5 * (1 + np.cos(np.pi * progress))

def create_efficientnet_emotion_model(dataset_name, learning_rate=0.001, total_epochs=50):
    """
    Creates an EfficientNetV2-based model for emotion detection.
    Uses transfer learning and fine-tuning for better performance.
    """
    num_classes = determine_num_classes(dataset_name)
    
    # Load pre-trained EfficientNetV2B0 without top layers
    base_model = EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_preprocessing=True
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create new model with improved grayscale handling
    inputs = layers.Input(shape=(48, 48, 1), name='original_input')
    
    # Enhanced grayscale to RGB conversion with learnable parameters
    conv1 = layers.Conv2D(3, (1, 1), name='grayscale_to_rgb')(inputs)
    
    # Progressive resizing with residual connections
    # First stage: 48x48 -> 96x96
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    residual = layers.Conv2D(32, (1, 1))(conv1)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    
    # Second stage: 96x96 -> 192x192
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    residual = layers.Conv2D(64, (1, 1))(layers.UpSampling2D(size=(2, 2))(residual))
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    
    # Final stage: 192x192 -> 224x224
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(3, (3, 3), padding='same')(x)  # Reduce to 3 channels for RGB
    x = layers.Activation('relu')(x)
    x = ResizeLayer((224, 224))(x)
    
    # Pass through EfficientNet
    x = base_model(x)
    
    # Enhanced top layers with residual connections
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.BatchNormalization()(x)
    
    # First dense block with residual
    dense1 = layers.Dense(512, activation=None)(x)
    x = layers.BatchNormalization()(dense1)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    residual = layers.Dense(512)(x)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    
    # Second dense block with residual
    dense2 = layers.Dense(256, activation=None)(x)
    x = layers.BatchNormalization()(dense2)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    residual = layers.Dense(256)(x)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs, name='EfficientNetV2_Emotion')
    
    # Custom learning rate scheduler with warmup
    lr_schedule = LearningRateScheduler(
        lambda epoch: cosine_decay_with_warmup(epoch, total_epochs, warmup_epochs=5, learning_rate_max=learning_rate)
    )
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience due to LR schedule
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # More aggressive reduction
            patience=3,   # Reduced patience
            min_lr=1e-7,
            verbose=1
        ),
        lr_schedule
    ]
    
    return model, callbacks

def unfreeze_and_train(model, num_layers_to_unfreeze=30):
    """
    Unfreezes the last few layers of the base model for fine-tuning.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.applications.EfficientNetV2B0):
            base_model = layer
            break
    
    if base_model is None:
        return model
        
    # Unfreeze the last num_layers_to_unfreeze layers
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    # Recompile with a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model 