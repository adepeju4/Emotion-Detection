import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ..config import (
    MODELS_SAVE_DIR, FER2013_EMOTIONS, CKPLUS_EMOTIONS,
    AFFECTNET_EMOTIONS, COMBINED_MODEL_EMOTIONS, EMOTION_CLASSES
)
from ..data_preprocessing.preprocess import load_fer2013, load_ckplus, load_affectnet, load_combined_dataset
from ..utils.metrics import compare_models
from ..utils.visualization import plot_confusion_matrix, visualize_predictions

from .train_fer2013 import train_fer2013_model
from .train_ckplus import train_ckplus_model
from .train_affectnet import train_affectnet_model
from .train_combined import train_combined_model

def train_all_models():
    """Train all emotion detection models and evaluate their performance."""
    try:
        
        print("Creating model save directory...")
        os.makedirs(MODELS_SAVE_DIR, exist_ok=True)
        
        # Set random seeds for reproducibility
        print("Setting random seeds...")
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Train all models
        print("\n=== Verifying Dataset Configurations ===")
        print(f"FER2013 emotions: {FER2013_EMOTIONS} ({len(FER2013_EMOTIONS)} classes)")
        print(f"CK+ emotions: {CKPLUS_EMOTIONS} ({len(CKPLUS_EMOTIONS)} classes)")
        print(f"AffectNet emotions: {AFFECTNET_EMOTIONS} ({len(AFFECTNET_EMOTIONS)} classes)")
        print(f"Combined emotions: {COMBINED_MODEL_EMOTIONS} ({len(COMBINED_MODEL_EMOTIONS)} classes)")
        
        # Train each model
        print("\n=== Training Models ===")
        models = {}
        
        print("\n-> Training FER2013 Model")
        models['FER2013'], _ = train_fer2013_model()
        
        print("\n-> Training CK+ Model")
        models['CK+'], _ = train_ckplus_model()
        
        print("\n-> Training AffectNet Model")
        models['AffectNet'], _ = train_affectnet_model()
        
        print("\n-> Training Combined Model")
        models['Combined'], _ = train_combined_model()
        
        # Load test datasets for evaluation
        print("\n=== Loading Test Datasets ===")
        test_datasets = {
            'FER2013': load_fer2013()[1],
            'CK+': load_ckplus()[1],
            'AffectNet': load_affectnet()[1],
            'Combined': load_combined_dataset()[1]
        }
        
        # Evaluate models
        print("\n=== Evaluating Models ===")
        results = compare_models(models, test_datasets, EMOTION_CLASSES)
        
        # Generate visualizations
        print("\n=== Generating Visualizations ===")
        os.makedirs('plots/confusion_matrices', exist_ok=True)
        os.makedirs('plots/sample_predictions', exist_ok=True)
        
        for model_name, model in models.items():
            for dataset_name, (X, y) in test_datasets.items():
                # Confusion matrices
                y_pred = model.predict(X)
                cm_fig = plot_confusion_matrix(y, y_pred, EMOTION_CLASSES)
                cm_fig.savefig(f'plots/confusion_matrices/{model_name}_{dataset_name}_confusion.png')
                plt.close(cm_fig)
                
                # Sample predictions
                pred_fig = visualize_predictions(model, X, y, EMOTION_CLASSES)
                pred_fig.savefig(f'plots/sample_predictions/{model_name}_{dataset_name}_samples.png')
                plt.close(pred_fig)
        
        print("\n=== Training and Evaluation Complete ===")
        return models, results
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("Starting model training and evaluation process...")
    models, results = train_all_models()
    if models and results:
        print("All models have been trained, evaluated, and saved successfully.") 