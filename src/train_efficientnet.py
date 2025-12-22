import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from architecture.efficientnet_architecture import create_efficientnet_emotion_model, unfreeze_and_train
from utils.metrics import plot_training_history, generate_classification_report
from utils.utils import load_dataset, setup_gpu_memory
from utils.data_analysis import analyze_dataset_distribution, verify_input_pipeline, analyze_model_preprocessing

def train_efficientnet(dataset_name='fer2013', epochs=50, batch_size=32):
    """
    Train the EfficientNetV2-based emotion detection model.
    Uses a two-phase training approach with enhanced preprocessing and analysis.
    """
    # Setup GPU memory
    setup_gpu_memory()
    
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_dataset(dataset_name)
    
    # Analyze dataset before preprocessing
    print("\nAnalyzing dataset distribution...")
    analyze_dataset_distribution(y_train, y_test, dataset_name)
    
    # Verify input pipeline
    print("\nVerifying input pipeline...")
    verify_input_pipeline(X_train, X_test)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # Create data generators with enhanced augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],  # Added brightness augmentation
        zoom_range=0.1,               # Added zoom augmentation
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator()
    
    # Create model with specified number of epochs for LR schedule
    model, callbacks = create_efficientnet_emotion_model(dataset_name, total_epochs=epochs)
    
    # Analyze model preprocessing on a sample batch
    print("\nAnalyzing model preprocessing...")
    analyze_model_preprocessing(model, X_train[:1])
    
    print("\nPhase 1: Training top layers...")
    # First phase: train only the top layers
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=test_datagen.flow(X_test, y_test, batch_size=batch_size),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save phase 1 results
    model.save(f'models/efficientnet_{dataset_name}_phase1.h5')
    plot_training_history(history, f'results/{dataset_name}/efficientnet_phase1_training_history.png')
    
    print("\nPhase 2: Fine-tuning...")
    # Second phase: fine-tune the last few layers
    model = unfreeze_and_train(model)
    
    # Train with a lower learning rate
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=test_datagen.flow(X_test, y_test, batch_size=batch_size),
        epochs=epochs // 2,  # Train for fewer epochs during fine-tuning
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model and results
    model.save(f'models/efficientnet_{dataset_name}_final.h5')
    plot_training_history(history, f'results/{dataset_name}/efficientnet_final_training_history.png')
    
    # Generate classification report
    y_pred = model.predict(X_test)
    generate_classification_report(y_test, y_pred, dataset_name, model_name='efficientnet')
    
    return model

if __name__ == '__main__':
    # Train on all datasets
    datasets = ['fer2013', 'ckplus', 'affectnet']
    
    for dataset in datasets:
        print(f"\nTraining EfficientNet model on {dataset} dataset...")
        train_efficientnet(dataset_name=dataset) 