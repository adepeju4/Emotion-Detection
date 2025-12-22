import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import os

def analyze_dataset_distribution(y_train, y_test, dataset_name):
    """Analyze and visualize the class distribution in the dataset"""
    # Convert one-hot encoded labels back to integers if needed
    if len(y_train.shape) > 1:
        y_train = np.argmax(y_train, axis=1)
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)
    
    # Get class counts
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    
    # Create labels for emotions
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    if len(train_counts) == 8:  # For FER2013 which has 'Contempt'
        emotions.insert(2, 'Contempt')
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(emotions))
    width = 0.35
    
    plt.bar(x - width/2, train_counts, width, label='Train')
    plt.bar(x + width/2, test_counts, width, label='Test')
    
    plt.xlabel('Emotion')
    plt.ylabel('Number of Samples')
    plt.title(f'Class Distribution in {dataset_name} Dataset')
    plt.xticks(x, emotions, rotation=45)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}/class_distribution.png')
    plt.close()
    
    # Print statistics
    print(f"\nDataset Statistics for {dataset_name}:")
    print("\nTraining Set:")
    for emotion, count in zip(emotions, train_counts):
        print(f"{emotion}: {count} samples ({count/len(y_train)*100:.2f}%)")
    
    print("\nTest Set:")
    for emotion, count in zip(emotions, test_counts):
        print(f"{emotion}: {count} samples ({count/len(y_test)*100:.2f}%)")

def verify_input_pipeline(X_train, X_test):
    """Verify the input pipeline and preprocessing"""
    # Check value ranges
    print("\nInput Pipeline Verification:")
    print(f"Training set - Min: {X_train.min():.2f}, Max: {X_train.max():.2f}, Mean: {X_train.mean():.2f}, Std: {X_train.std():.2f}")
    print(f"Test set - Min: {X_test.min():.2f}, Max: {X_test.max():.2f}, Mean: {X_test.mean():.2f}, Std: {X_test.std():.2f}")
    
    # Check for NaN or infinity values
    print(f"\nNaN values in training set: {np.isnan(X_train).any()}")
    print(f"Infinity values in training set: {np.isinf(X_train).any()}")
    print(f"NaN values in test set: {np.isnan(X_test).any()}")
    print(f"Infinity values in test set: {np.isinf(X_test).any()}")
    
    # Ensure directory exists
    os.makedirs('results', exist_ok=True)
    
    # Visualize sample images and their preprocessed versions
    plt.figure(figsize=(15, 5))
    for i in range(5):
        # Original image
        plt.subplot(2, 5, i+1)
        plt.imshow(X_train[i].squeeze(), cmap='gray')
        plt.title(f'Original {i+1}')
        plt.axis('off')
        
        # Convert numpy array to tensor and add batch dimension
        img_tensor = tf.convert_to_tensor(X_train[i:i+1])
        if len(img_tensor.shape) == 3:  # Add channel dimension if needed
            img_tensor = tf.expand_dims(img_tensor, -1)
        
        # Preprocessed image (after grayscale to RGB conversion)
        plt.subplot(2, 5, i+6)
        rgb_img = tf.image.grayscale_to_rgb(img_tensor)
        rgb_img = tf.image.resize(rgb_img, (224, 224))
        plt.imshow(rgb_img[0].numpy())
        plt.title(f'Preprocessed {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/preprocessing_verification.png')
    plt.close()

def analyze_model_preprocessing(model, X_sample):
    """Analyze the model's preprocessing layers"""
    # Convert input to tensor
    X_tensor = tf.convert_to_tensor(X_sample)
    if len(X_tensor.shape) == 3:  # Add channel dimension if needed
        X_tensor = tf.expand_dims(X_tensor, -1)
    
    # Get the output of the grayscale_to_rgb layer
    rgb_conv = model.get_layer('grayscale_to_rgb')
    rgb_output = rgb_conv(X_tensor)
    
    # Get outputs after each upsampling stage
    stage1_output = None
    stage2_output = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.UpSampling2D):
            if stage1_output is None:
                stage1_output = layer(rgb_output)
            else:
                stage2_output = layer(stage1_output)
                break
    
    # Visualize the transformations
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(141)
    plt.imshow(X_sample[0].squeeze(), cmap='gray')
    plt.title('Original (48x48)')
    plt.axis('off')
    
    # After RGB conversion
    plt.subplot(142)
    plt.imshow(rgb_output[0].numpy())
    plt.title('After RGB Conv')
    plt.axis('off')
    
    # After first upsampling
    plt.subplot(143)
    plt.imshow(stage1_output[0].numpy())
    plt.title('After First Upsample')
    plt.axis('off')
    
    # After second upsampling
    plt.subplot(144)
    plt.imshow(stage2_output[0].numpy())
    plt.title('Final (224x224)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/preprocessing_stages.png')
    plt.close() 