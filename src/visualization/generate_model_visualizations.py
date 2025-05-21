import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model architectures
from architecture.cnn_architecture import create_emotion_model as create_basic_cnn
from architecture.cnn_architecture_improved import create_improved_emotion_model as create_improved_cnn

def save_model_architecture(model_name, save_dir):
    """Save model architecture visualization"""
    try:
        # Pass the model name to let the architecture determine the number of classes
        if model_name.lower() == 'ckplus':
            model = create_basic_cnn(dataset_name=model_name.lower())
        else:
            model, _ = create_improved_cnn(dataset_name=model_name.lower())  # Ignore callbacks
        
        # Save architecture diagram
        tf.keras.utils.plot_model(
            model,
            to_file=os.path.join(save_dir, f'{model_name.lower()}_architecture.png'),
            show_shapes=True,
            show_layer_names=True,
            dpi=200
        )
        print(f"Generated architecture visualization for {model_name}")
    except Exception as e:
        print(f"Could not generate architecture visualization for {model_name}: {str(e)}")

def main():
    # Create output directory
    save_dir = 'results/model_visualizations'
    os.makedirs(save_dir, exist_ok=True)
    
    # Models to process with correct dataset names
    models = ['affectnet', 'ckplus', 'fer2013']
    
    for model_name in models:
        save_model_architecture(model_name, save_dir)

if __name__ == '__main__':
    main() 