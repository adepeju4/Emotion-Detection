import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from .visualization import plot_all_datasets_distribution, visualize_feature_maps
from data_preprocessing.preprocess import load_fer2013
import config

def analyze_dataset_distributions():
    """Analyze and visualize the distribution of emotions across all datasets."""
    print("Analyzing emotion distributions across all datasets...")
    plot_all_datasets_distribution()
    print("Analysis complete! Check the 'plots' directory for results.")

def analyze_model_features(model_path=config.FER2013_MODEL_PATH, layer_names=None):
    """
    Analyze and visualize feature maps from a trained model.
    
    Args:
        model_path: Path to the trained model
        layer_names: List of layer names to visualize. If None, uses default layers.
    """
    print("Loading trained model...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading sample images...")
    (_, _), (test_X, _) = load_fer2013()
    sample_img = test_X[0]
    
    # Default layers to visualize if none specified
    if layer_names is None:
        layer_names = [
            'conv2d',  # First conv layer
            'max_pooling2d',  # First pooling layer
            'conv2d_1',  # Second conv layer
        ]
    
    print("Generating feature map visualizations...")
    for layer_name in layer_names:
        print(f"Visualizing layer: {layer_name}")
        visualize_feature_maps(model, sample_img, layer_name)
    
    print("Analysis complete! Check the 'plots/feature_maps' directory for results.")

def analyze_model_performance(model_path=config.FER2013_MODEL_PATH):
    """
    Analyze model performance metrics and generate visualizations.
    
    Args:
        model_path: Path to the trained model
    """
    print("Loading trained model...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading test data...")
    (_, _), (test_X, test_y) = load_fer2013()
    
    print("Evaluating model...")
    results = model.evaluate(test_X, test_y, verbose=1)
    metrics = dict(zip(model.metrics_names, results))
    
    print("\nModel Performance Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    return metrics

def run_full_analysis(model_path=config.FER2013_MODEL_PATH):
    """
    Run a comprehensive analysis including dataset distributions,
    model features, and performance metrics.
    
    Args:
        model_path: Path to the trained model
    """
    print("\n=== Starting Full Analysis ===\n")
    
    print("1. Analyzing Dataset Distributions")
    print("---------------------------------")
    analyze_dataset_distributions()
    
    print("\n2. Analyzing Model Features")
    print("---------------------------")
    analyze_model_features(model_path)
    
    print("\n3. Analyzing Model Performance")
    print("------------------------------")
    metrics = analyze_model_performance(model_path)
    
    print("\n=== Analysis Complete ===")
    return metrics 