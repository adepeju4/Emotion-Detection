from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf

def calculate_metrics(y_true, y_pred):
    """Calculate various metrics for model evaluation."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics 

def plot_training_history(history, save_path):
    """
    Plot training history including accuracy and loss curves.
    
    Args:
        history: Keras history object
        save_path: Path to save the plot
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_classification_report(y_true, y_pred, dataset_name, model_name='efficientnet'):
    """
    Generate and save classification metrics including confusion matrix.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted labels (one-hot encoded)
        dataset_name: Name of the dataset
        model_name: Name of the model
    """
    # Convert one-hot encoded labels back to class indices
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Create directory for results
    results_dir = f'results/{dataset_name}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save classification report
    report = classification_report(y_true_classes, y_pred_classes)
    with open(f'{results_dir}/{model_name}_classification_report.txt', 'w') as f:
        f.write(f'Model: {model_name}\n')
        f.write(f'Dataset: {dataset_name}\n')
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'F1 Score (weighted): {f1:.4f}\n\n')
        f.write('Classification Report:\n')
        f.write(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{results_dir}/{model_name}_confusion_matrix.png')
    plt.close()

def plot_learning_rate(history, save_path):
    """
    Plot learning rate changes during training.
    
    Args:
        history: Keras history object
        save_path: Path to save the plot
    """
    if 'lr' not in history.history:
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['lr'])
    plt.title('Learning Rate over Time')
    plt.ylabel('Learning Rate')
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_feature_maps(model, image, layer_name, save_path, max_features=16):
    """
    Plot feature maps from a specific layer for a given image.
    
    Args:
        model: Keras model
        image: Input image (should be preprocessed)
        layer_name: Name of the layer to visualize
        save_path: Path to save the plot
        max_features: Maximum number of feature maps to plot
    """
    # Get the output of the specified layer
    layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    feature_maps = layer_model.predict(image)
    
    # Get number of features to plot
    n_features = min(max_features, feature_maps.shape[-1])
    size = int(np.ceil(np.sqrt(n_features)))
    
    # Create figure
    plt.figure(figsize=(20, 20))
    for i in range(n_features):
        plt.subplot(size, size, i + 1)
        plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
        plt.axis('off')
    
    plt.suptitle(f'Feature Maps - {layer_name}')
    plt.savefig(save_path)
    plt.close() 