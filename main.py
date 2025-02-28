import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("Starting emotion detection project...")

try:
    print("Importing config...")
    import config
    print("Importing data preprocessing modules...")
    from data_preprocessing.preprocess import load_fer2013, load_ckplus, load_affectnet, load_combined_dataset
    print("Importing model modules...")
    from models.cnn_model import create_emotion_model, create_lightweight_model
    print("Importing training modules...")
    from train.train_fer2013 import train_fer2013_model
    from train.train_ckplus import train_ckplus_model
    from train.train_affectnet import train_affectnet_model
    from train.train_combined import train_combined_model
    print("Importing utility modules...")
    from utils.metrics import evaluate_model, compare_models
    from utils.visualization import plot_confusion_matrix, visualize_predictions, plot_all_datasets_distribution
    
    print("All modules imported successfully.")
except Exception as e:
    print(f"Error during imports: {e}")
    import traceback
    traceback.print_exc()

def main():
    """Main function to run the emotion detection project"""
    print("Entered main function")
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    try:
        # Create directories if they don't exist
        print("Creating model save directory...")
        os.makedirs(config.MODELS_SAVE_DIR, exist_ok=True)
        
        # Set random seeds for reproducibility
        print("Setting random seeds...")
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Train models or load pre-trained models
        train_new_models = True
        
        if train_new_models:
            print("\n=== Skipping Dataset Distributions Plotting ===")
            # plot_all_datasets_distribution()  # Uncomment when visualization.py is updated
            
            print("\n=== Verifying Dataset Configurations ===")
            print(f"FER2013 emotions: {config.FER2013_EMOTIONS} ({len(config.FER2013_EMOTIONS)} classes)")
            print(f"CK+ emotions: {config.CKPLUS_EMOTIONS} ({len(config.CKPLUS_EMOTIONS)} classes)")
            print(f"AffectNet emotions: {config.AFFECTNET_EMOTIONS} ({len(config.AFFECTNET_EMOTIONS)} classes)")
            print(f"Combined emotions: {config.COMBINED_MODEL_EMOTIONS} ({len(config.COMBINED_MODEL_EMOTIONS)} classes)")
            
            print("\n=== Training FER2013 Model ===")
            fer2013_model, _ = train_fer2013_model()
            
            print("\n=== Training CK+ Model ===")
            ckplus_model, _ = train_ckplus_model()
            
            print("\n=== Training AffectNet Model ===")
            affectnet_model, _ = train_affectnet_model()
            
            print("\n=== Training Combined Model ===")
            combined_model, _ = train_combined_model()
        else:
            # Load pre-trained models
            print("Loading pre-trained models...")
            fer2013_model = tf.keras.models.load_model(config.FER2013_MODEL_PATH)
            ckplus_model = tf.keras.models.load_model(config.CKPLUS_MODEL_PATH)
            affectnet_model = tf.keras.models.load_model(config.AFFECTNET_MODEL_PATH)
            combined_model = tf.keras.models.load_model(config.COMBINED_MODEL_PATH)
        
        # Load test data for evaluation
        print("Loading test data for evaluation...")
        (_, _), (fer2013_test_X, fer2013_test_y) = load_fer2013()
        (_, _), (ckplus_test_X, ckplus_test_y) = load_ckplus()
        (_, _), (affectnet_test_X, affectnet_test_y) = load_affectnet()
        (_, _), (combined_test_X, combined_test_y) = load_combined_dataset()
        
        # Create dictionary of models and test datasets
        models = {
            'FER2013': fer2013_model,
            'CK+': ckplus_model,
            'AffectNet': affectnet_model,
            'Combined': combined_model
        }
        
        test_datasets = {
            'FER2013': (fer2013_test_X, fer2013_test_y),
            'CK+': (ckplus_test_X, ckplus_test_y),
            'AffectNet': (affectnet_test_X, affectnet_test_y),
            'Combined': (combined_test_X, combined_test_y)
        }
        
        # Compare models
        print("\n=== Model Comparison ===")
        results = compare_models(models, test_datasets, config.EMOTION_CLASSES)
        
        # Visualize confusion matrices
        print("\n=== Generating Confusion Matrices ===")
        for model_name, model in models.items():
            for dataset_name, (X, y) in test_datasets.items():
                y_pred = model.predict(X)
                cm_fig = plot_confusion_matrix(y, y_pred, config.EMOTION_CLASSES)
                cm_fig.savefig(f'confusion_matrix_{model_name}_{dataset_name}.png')
                plt.close(cm_fig)
        
        # Visualize sample predictions
        print("\n=== Generating Sample Predictions ===")
        for model_name, model in models.items():
            for dataset_name, (X, y) in test_datasets.items():
                pred_fig = visualize_predictions(model, X, y, config.EMOTION_CLASSES)
                pred_fig.savefig(f'sample_predictions_{model_name}_{dataset_name}.png')
                plt.close(pred_fig)
        
        print("\n=== Emotion Detection Project Complete ===")
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

print("Calling main function...")
if __name__ == "__main__":
    main()
    print("Main function completed.") 