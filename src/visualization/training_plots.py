import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Model

def ensure_result_dir(model_name):
    result_dir = os.path.join('results', model_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

def plot_training_history(history, model_name): 
    result_dir = ensure_result_dir(model_name)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'training_history.png'))
    plt.close()

def plot_learning_rate(history, model_name):
    if 'lr' in history.history:
        result_dir = ensure_result_dir(model_name)
        plt.figure(figsize=(10, 6))
        
       
        lr_values = []
        for logs in history.history.get('lr', []):
            lr_values.append(logs)
        
        plt.plot(lr_values, marker='o')
        plt.title('Learning Rate During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log') 
        plt.grid(True)
        
        plt.savefig(os.path.join(result_dir, 'learning_rate.png'))
        plt.close()

def plot_per_class_accuracy(y_true, y_pred, class_names, model_name):
    result_dir = ensure_result_dir(model_name)
    class_accuracy = []
    for i in range(len(class_names)):
        mask = (y_true.argmax(axis=1) == i)
        class_acc = (y_pred.argmax(axis=1)[mask] == i).mean()
        class_accuracy.append(class_acc)
    
    plt.figure(figsize=(10, 5))
    plt.bar(class_names, class_accuracy)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Emotion Class')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'per_class_accuracy.png'))
    plt.close()

def plot_classification_report(y_true, y_pred, class_names, model_name):
    result_dir = ensure_result_dir(model_name)
    report = classification_report(y_true.argmax(axis=1), 
                                 y_pred.argmax(axis=1), 
                                 target_names=class_names)
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    plt.text(0.1, 0.1, report, fontsize=12, family='monospace')
    plt.title('Classification Report')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'classification_report.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    result_dir = ensure_result_dir(model_name)
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))
    plt.close()

def visualize_feature_maps(model, test_image, layer_names=None, model_name=None):
    result_dir = ensure_result_dir(model_name)
    if layer_names is None:
        layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]
    
    for layer_name in layer_names:
        feature_model = Model(inputs=model.input,
                            outputs=model.get_layer(layer_name).output)
        
        feature_maps = feature_model.predict(test_image[np.newaxis, ...])
        
        n_features = min(16, feature_maps.shape[-1])
        size = int(np.ceil(np.sqrt(n_features)))
        
        plt.figure(figsize=(20, 20))
        for i in range(n_features):
            plt.subplot(size, size, i + 1)
            plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
            plt.axis('off')
        
        plt.suptitle(f'Feature Maps of {layer_name}')
        plt.savefig(os.path.join(result_dir, f'feature_maps_{layer_name}.png'))
        plt.close()

def plot_sample_predictions(model, X_test, y_test, class_names, model_name, num_samples=5):
    result_dir = ensure_result_dir(model_name)
    predictions = model.predict(X_test)
    
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X_test[i].reshape(48, 48), cmap='gray')
        true_class = class_names[y_test[i].argmax()]
        pred_class = class_names[predictions[i].argmax()]
        plt.title(f'True: {true_class}\nPred: {pred_class}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'sample_predictions.png'))
    plt.close()

def create_all_visualizations(model, history, X_test, y_test, class_names, model_name):
    y_pred = model.predict(X_test)
    
    plot_training_history(history, model_name)
    plot_learning_rate(history, model_name)
    plot_per_class_accuracy(y_test, y_pred, class_names, model_name)
    plot_classification_report(y_test, y_pred, class_names, model_name)
    plot_confusion_matrix(y_test, y_pred, class_names, model_name)
    visualize_feature_maps(model, X_test[0], model_name=model_name)
    plot_sample_predictions(model, X_test, y_test, class_names, model_name) 