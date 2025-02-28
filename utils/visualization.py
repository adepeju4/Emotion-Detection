import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import cv2
import config

def plot_confusion_matrix(y_true, y_pred, class_names=config.EMOTION_CLASSES):
    """Plot confusion matrix for model evaluation"""
    # Convert one-hot encoded labels to class indices
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    return plt.gcf()

def visualize_predictions(model, X, y_true, class_names=config.EMOTION_CLASSES, num_samples=5):
    """Visualize model predictions on sample images"""
    # Get predictions
    y_pred = model.predict(X)
    
    # Convert one-hot encoded labels to class indices
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true
    
    y_pred_indices = np.argmax(y_pred, axis=1)
    
    # Randomly select samples
    indices = np.random.choice(range(len(X)), size=min(num_samples, len(X)), replace=False)
    
    # Create figure
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        # Get image
        img = X[idx].squeeze()
        
        # Get true and predicted labels
        true_label = class_names[y_true_indices[idx]]
        pred_label = class_names[y_pred_indices[idx]]
        pred_confidence = y_pred[idx][y_pred_indices[idx]] * 100
        
        # Display image
        if len(axes.shape) == 0:
            ax = axes
        else:
            ax = axes[i]
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {pred_confidence:.1f}%")
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def visualize_feature_maps(model, img, layer_name=None):
    """Visualize feature maps from a specific layer for an input image"""
    # Preprocess image if needed
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    
    # If no layer specified, use the first convolutional layer
    if layer_name is None:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    
    # Create a model that outputs the feature maps
    feature_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    
    # Get feature maps
    feature_maps = feature_model.predict(img)
    
    # Plot feature maps
    feature_maps = feature_maps[0]
    n_features = feature_maps.shape[-1]
    size = feature_maps.shape[0]
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(n_features)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    
    # Plot each feature map
    for i in range(grid_size):
        for j in range(grid_size):
            feature_idx = i * grid_size + j
            if feature_idx < n_features:
                if grid_size == 1:
                    ax = axes
                else:
                    ax = axes[i, j]
                ax.imshow(feature_maps[:, :, feature_idx], cmap='viridis')
                ax.set_title(f'Feature {feature_idx + 1}')
            ax.axis('off')
    
    plt.suptitle(f'Feature Maps for Layer: {layer_name}')
    plt.tight_layout()
    return fig 