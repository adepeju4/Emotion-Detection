# webcam_emotion_detection.py
import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import argparse

# Add the project root to the path so we can import config
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config

# Print emotion classes for debugging
print(f"Emotion classes from config: {config.EMOTION_CLASSES}")

# Define colors for all possible emotions (BGR format)
EMOTION_COLORS = {
    'anger': (0, 0, 255),       # Red
    'angry': (0, 0, 255),       # Red (alternative name)
    'contempt': (120, 0, 120),  # Purple
    'disgust': (0, 140, 0),     # Dark Green
    'fear': (255, 0, 255),      # Magenta
    'happiness': (0, 255, 255), # Yellow
    'happy': (0, 255, 255),     # Yellow (alternative name)
    'sadness': (255, 0, 0),     # Blue
    'sad': (255, 0, 0),         # Blue (alternative name)
    'surprise': (0, 255, 0),    # Green
    'neutral': (128, 128, 128)  # Grey
}

# Choose which model to use: 'fer2013', 'ckplus', 'affectnet', or 'combined'
SELECTED_MODEL = 'ckplus'

# At the top of your script
DEBUG_MODE = True

def preprocess_frame(frame, target_size=(config.IMG_SIZE, config.IMG_SIZE)):
    """Preprocess a webcam frame for prediction with improved face detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate minimum face size (as a percentage of the frame)
    frame_height, frame_width = frame.shape[:2]
    min_face_size = int(min(frame_height, frame_width) * 0.2)  # At least 20% of frame
    
    # Use a face detector with better parameters
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,       
        minNeighbors=8,        # Increased from 5 to reduce false positives
        minSize=(min_face_size, min_face_size),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Filter out overlapping faces
    filtered_faces = []
    for i, face_i in enumerate(faces):
        x_i, y_i, w_i, h_i = face_i
        area_i = w_i * h_i
        keep = True
        
        for j, face_j in enumerate(faces):
            if i == j:
                continue
                
            x_j, y_j, w_j, h_j = face_j
            
            # Check if faces overlap significantly
            overlap_x = max(0, min(x_i + w_i, x_j + w_j) - max(x_i, x_j))
            overlap_y = max(0, min(y_i + h_i, y_j + h_j) - max(y_i, y_j))
            overlap_area = overlap_x * overlap_y
            
            # If there's significant overlap and the other face is larger
            if overlap_area > 0.3 * area_i and w_j * h_j > area_i:
                keep = False
                break
                
        if keep:
            filtered_faces.append(face_i)
    
    # Replace original faces with filtered ones
    faces = filtered_faces
    
    processed_faces = []
    face_locations = []
    
    for (x, y, w, h) in faces:
        # Skip faces that are too small
        if w < min_face_size or h < min_face_size:
            continue
            
        # Extract face
        face = gray[y:y+h, x:x+w]
        
        # Resize
        face = cv2.resize(face, target_size)
        
        # Normalize
        face = face / 255.0
        
        # Reshape for model input
        face = np.expand_dims(face, axis=-1)  # Add channel dimension
        
        processed_faces.append(face)
        face_locations.append((x, y, w, h))
    
    if processed_faces:
        return np.array(processed_faces), face_locations
    else:
        return None, []

def create_emotion_bar_chart(figsize=(5, 6), emotion_classes=None):
    """Create a matplotlib figure for the vertical emotion probability bar chart"""
    # Use provided emotion classes or default to config
    if emotion_classes is None:
        emotion_classes = config.EMOTION_CLASSES
        
    fig = plt.figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)
    
    # Set up the chart
    ax.set_title('Emotion Probabilities')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    
    # Initialize empty vertical bar chart
    x = np.arange(len(emotion_classes))
    bars = ax.bar(x, np.zeros(len(emotion_classes)), width=0.7)
    
    # Set colors for bars
    for i, emotion in enumerate(emotion_classes):
        color = EMOTION_COLORS.get(emotion, (128, 128, 128))  # Default to gray if not found
        # Convert BGR to RGB for matplotlib
        color_rgb = (color[2]/255, color[1]/255, color[0]/255)
        bars[i].set_color(color_rgb)
    
    # Set x-axis labels
    plt.xticks(x, emotion_classes, rotation=45, ha='right', fontsize=10)
    
    # Add value labels above the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 0.01,
                '0%', ha='center', va='bottom', rotation=90, fontsize=10)
    
    # Add padding
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.3)
    
    return fig, ax, bars, emotion_classes

def update_emotion_bar_chart(fig, ax, bars, prediction):
    """Update the emotion probability bar chart with new predictions"""
    # Update bar heights
    for i, bar in enumerate(bars):
        bar.set_height(prediction[i])
        
        # Update text labels
        ax.texts[i].set_text(f'{prediction[i]*100:.1f}%')
        ax.texts[i].set_position((bar.get_x() + bar.get_width()/2., max(0.01, prediction[i] + 0.02)))
        
        # Adjust text color and rotation based on bar height
        if prediction[i] > 0.1:
            ax.texts[i].set_rotation(0)  # Horizontal text for taller bars
            ax.texts[i].set_va('bottom')
        else:
            ax.texts[i].set_rotation(90)  # Vertical text for shorter bars
            ax.texts[i].set_va('bottom')
    
    # Convert matplotlib figure to OpenCV image
    canvas = FigureCanvas(fig)
    canvas.draw()
    chart_image = np.array(canvas.renderer.buffer_rgba())
    chart_image = cv2.cvtColor(chart_image, cv2.COLOR_RGBA2BGR)
    
    return chart_image

def run_webcam_detection():
    """Run real-time emotion detection on webcam feed with multiple models"""
    # Create model manager
    model_manager = EmotionModelManager()
    
    # Load all models with EXACT emotion orders from your datasets
    # FER2013: Based on your dataset structure
    model_manager.load_model(
        'fer2013', 
        config.FER2013_MODEL_PATH, 
        ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    )
    
    # CK+: Based on your dataset structure
    model_manager.load_model(
        'ckplus', 
        config.CKPLUS_MODEL_PATH, 
        ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    )
    
    # AffectNet: Based on your dataset structure
    model_manager.load_model(
        'affectnet', 
        config.AFFECTNET_MODEL_PATH, 
        ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    )
    
    # Set initial active model
    active_model = 'fer2013'  # Default
    model_manager.set_active_model(active_model)
    
    # Start webcam
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get webcam dimensions
    webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create emotion bar charts for each model
    charts = {}
    for model_name in model_manager.models.keys():
        emotion_classes = model_manager.emotion_classes[model_name]
        fig, ax, bars, _ = create_emotion_bar_chart(
            figsize=(4, 3),
            emotion_classes=emotion_classes
        )
        charts[model_name] = {
            'fig': fig,
            'ax': ax,
            'bars': bars,
            'image': None
        }
    
    # Create a combined display frame (webcam feed + sidebar)
    sidebar_width = 400  # Width of the sidebar in pixels
    combined_width = webcam_width + sidebar_width
    combined_height = webcam_height
    
    print("Webcam started. Press 'q' to quit, '1' for FER2013, '2' for CK+, '3' for AffectNet")
    
    # For tracking FPS
    frame_count = 0
    start_time = cv2.getTickCount()
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Preprocess frame for prediction
        faces, face_locations = preprocess_frame(frame)
        
        # Create combined frame (webcam + sidebar)
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Place webcam feed on the left
        combined_frame[0:webcam_height, 0:webcam_width] = display_frame
        
        # Create sidebar (black background)
        sidebar_padding = 10
        
        # Process faces if detected
        if faces is not None and len(faces) > 0:
            # Get predictions from all models
            all_predictions = model_manager.predict(faces)
            
            # Update charts for all models
            for model_name, (predictions, emotion_classes) in all_predictions.items():
                chart = charts[model_name]
                chart['image'] = update_emotion_bar_chart(chart['fig'], chart['ax'], chart['bars'], predictions[0])
            
            # Process active model predictions
            active_predictions, active_emotion_classes = all_predictions[active_model]
            
            # Draw rectangles and labels for each face using active model
            for i, (x, y, w, h) in enumerate(face_locations):
                # Get the predicted emotion for this face
                prediction = active_predictions[i]
                emotion_idx = np.argmax(prediction)
                emotion = active_emotion_classes[emotion_idx]
                confidence = prediction[emotion_idx] * 100
                
                # Get color for the predicted emotion
                color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                
                # Draw rectangle around face
                cv2.rectangle(combined_frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label background
                label_bg_color = (*color, 0.7)  # Add alpha for transparency
                label_text = f"{emotion} ({confidence:.1f}%)"
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(combined_frame, (x, y-text_height-10), (x+text_width+10, y), color, -1)
                
                # Draw label text
                cv2.putText(combined_frame, label_text, (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Update frame count for FPS calculation
        frame_count += 1
        
        # Calculate and display FPS
        if frame_count >= 10:  # Update FPS every 10 frames
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            fps = frame_count / elapsed_time
            
            # Reset counters
            frame_count = 0
            start_time = current_time
            
            # Display FPS
            cv2.putText(combined_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        
        # Display active model name
        cv2.putText(combined_frame, f"Active Model: {active_model}", 
                   (webcam_width + sidebar_padding, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Add charts for all models to the sidebar
        y_offset = 60
        for model_name, chart_data in charts.items():
            if chart_data['image'] is not None:
                # Resize chart to fit in the sidebar
                chart_height = 150
                resized_chart = cv2.resize(chart_data['image'], 
                                         (sidebar_width - sidebar_padding * 2, chart_height))
                
                # Place the chart in the sidebar
                combined_frame[y_offset:y_offset+chart_height, 
                             webcam_width+sidebar_padding:webcam_width+sidebar_width-sidebar_padding] = resized_chart
                
                # Add title above the chart
                cv2.putText(combined_frame, f"{model_name} Emotions", 
                           (webcam_width + sidebar_padding + 10, y_offset - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Highlight active model
                if model_name == active_model:
                    cv2.rectangle(combined_frame, 
                                 (webcam_width+5, y_offset-30), 
                                 (webcam_width+sidebar_width-5, y_offset+chart_height+5), 
                                 (0, 255, 0), 2)
                
                y_offset += chart_height + 30
        
        # Display the combined frame
        cv2.imshow('Emotion Detection', combined_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            active_model = 'fer2013'
            model_manager.set_active_model(active_model)
        elif key == ord('2'):
            active_model = 'ckplus'
            model_manager.set_active_model(active_model)
        elif key == ord('3'):
            active_model = 'affectnet'
            model_manager.set_active_model(active_model)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    for chart_data in charts.values():
        plt.close(chart_data['fig'])
    print("Webcam closed.")

    if DEBUG_MODE:
        # Print raw prediction values
        print("\nRaw prediction values:")
        for i, emotion in enumerate(active_emotion_classes):
            print(f"{emotion}: {active_predictions[i]:.6f}")

class EmotionModelManager:
    """Class to manage multiple emotion detection models"""
    
    def __init__(self):
        self.models = {}
        self.emotion_classes = {}
        self.active_model = None
    
    def load_model(self, model_name, model_path, emotion_classes):
        """Load a model and its emotion classes"""
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return False
        
        try:
            print(f"Loading {model_name} model...")
            model = tf.keras.models.load_model(model_path)
            
            # Print model output shape for debugging
            print(f"Model output shape: {model.output_shape}")
            num_classes = model.output_shape[-1]
            print(f"Number of output classes: {num_classes}")
            
            # Check if the number of classes matches
            if num_classes != len(emotion_classes):
                print(f"WARNING: Model has {num_classes} outputs but we're using {len(emotion_classes)} emotion classes!")
                # Adjust emotion classes if needed
                if model_name == 'ckplus' and num_classes == 6:
                    # Remove contempt for 6-class CK+
                    emotion_classes = [e for e in emotion_classes if e != 'contempt']
                    print(f"Adjusted to 6-class CK+ emotions: {emotion_classes}")
            
            # Store the model and its emotion classes
            self.models[model_name] = model
            self.emotion_classes[model_name] = emotion_classes
            
            # Set as active model if it's the first one
            if not self.active_model:
                self.active_model = model_name
                
            return True
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return False
    
    def predict(self, face_input, model_name=None):
        """Make a prediction using the specified model or all models"""
        if model_name and model_name in self.models:
            # Predict with specific model
            model = self.models[model_name]
            prediction = model.predict(face_input, verbose=0)
            return {model_name: (prediction, self.emotion_classes[model_name])}
        else:
            # Predict with all models
            results = {}
            for name, model in self.models.items():
                prediction = model.predict(face_input, verbose=0)
                results[name] = (prediction, self.emotion_classes[name])
            return results
    
    def set_active_model(self, model_name):
        """Set the active model for display"""
        if model_name in self.models:
            self.active_model = model_name
            return True
        return False
    
    def get_active_model(self):
        """Get the active model name"""
        return self.active_model

def run_single_model_detection(model_name):
    """Run webcam detection with a single model"""
    # Set model path and emotion classes based on selected model
    if model_name == 'fer2013':
        model_path = config.FER2013_MODEL_PATH
        emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    elif model_name == 'ckplus':
        model_path = config.CKPLUS_MODEL_PATH
        emotion_classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    elif model_name == 'affectnet':
        model_path = config.AFFECTNET_MODEL_PATH
        emotion_classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    elif model_name == 'combined':
        model_path = config.COMBINED_MODEL_PATH
        emotion_classes = config.COMBINED_MODEL_EMOTIONS
    else:
        # Default to FER2013
        model_path = config.FER2013_MODEL_PATH
        emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first or specify the correct model path.")
        return
    
    # Load model
    print(f"Loading {model_name} model...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Print model output shape for debugging
    print(f"Model output shape: {model.output_shape}")
    num_classes = model.output_shape[-1]
    print(f"Number of output classes: {num_classes}")
    
    # Check if the number of classes matches
    if num_classes != len(emotion_classes):
        print(f"WARNING: Model has {num_classes} outputs but we're using {len(emotion_classes)} emotion classes!")
        # Adjust emotion classes if needed
        if model_name == 'ckplus' and num_classes == 6:
            # Remove contempt for 6-class CK+
            emotion_classes = [e for e in emotion_classes if e != 'contempt']
            print(f"Adjusted to 6-class CK+ emotions: {emotion_classes}")
    
    print(f"Using emotion classes: {emotion_classes}")
    
    # Create emotion bar chart
    fig, ax, bars, _ = create_emotion_bar_chart(
        figsize=(4, 3),
        emotion_classes=emotion_classes
    )
    
    # Start webcam
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get webcam dimensions
    webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a combined display frame (webcam feed + sidebar)
    sidebar_width = 400  # Width of the sidebar in pixels
    combined_width = webcam_width + sidebar_width
    combined_height = webcam_height
    
    # For tracking FPS
    frame_count = 0
    start_time = cv2.getTickCount()
    
    # For storing chart image
    chart_image = None
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Preprocess frame for prediction
        faces, face_locations = preprocess_frame(frame)
        
        # Create combined frame (webcam + sidebar)
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Place webcam feed on the left
        combined_frame[0:webcam_height, 0:webcam_width] = display_frame
        
        # Create sidebar (black background)
        sidebar_padding = 10
        
        # Process faces if detected
        if faces is not None and len(faces) > 0:
            # Make predictions
            predictions = model.predict(faces, verbose=0)
            
            # Update chart
            chart_image = update_emotion_bar_chart(fig, ax, bars, predictions[0])
            
            # Draw rectangles and labels for each face
            for i, (x, y, w, h) in enumerate(face_locations):
                # Get the predicted emotion for this face
                prediction = predictions[i]
                emotion_idx = np.argmax(prediction)
                emotion = emotion_classes[emotion_idx]
                confidence = prediction[emotion_idx] * 100
                
                # Get color for the predicted emotion
                color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                
                # Draw rectangle around face
                cv2.rectangle(combined_frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label background
                label_text = f"{emotion} ({confidence:.1f}%)"
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(combined_frame, (x, y-text_height-10), (x+text_width+10, y), color, -1)
                
                # Draw label text
                cv2.putText(combined_frame, label_text, (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Update frame count for FPS calculation
        frame_count += 1
        
        # Calculate and display FPS
        if frame_count >= 10:  # Update FPS every 10 frames
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            fps = frame_count / elapsed_time
            
            # Reset counters
            frame_count = 0
            start_time = current_time
            
            # Display FPS
            cv2.putText(combined_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        
        # Display model name
        cv2.putText(combined_frame, f"Model: {model_name}", 
                   (webcam_width + sidebar_padding, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Add emotion chart to the sidebar if available
        if chart_image is not None:
            # Calculate position for chart
            chart_height = 300
            chart_y = (combined_height - chart_height) // 2
            
            # Resize chart to fit in the sidebar
            resized_chart = cv2.resize(chart_image, 
                                     (sidebar_width - sidebar_padding * 2, chart_height))
            
            # Place the chart in the sidebar
            combined_frame[chart_y:chart_y+chart_height, 
                         webcam_width+sidebar_padding:webcam_width+sidebar_width-sidebar_padding] = resized_chart
            
            # Add title above the chart
            cv2.putText(combined_frame, "Detected Emotions", 
                       (webcam_width + sidebar_padding + 10, chart_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Display the combined frame
        cv2.imshow('Emotion Detection', combined_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    plt.close(fig)
    print("Webcam closed.")
    
    if DEBUG_MODE:
        # Print raw prediction values
        print("\nRaw prediction values:")
        for i, emotion in enumerate(emotion_classes):
            print(f"{emotion}: {predictions[0][i]:.6f}")

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Emotion Detection Webcam Demo')
    parser.add_argument('--model', type=str, default='fer2013', 
                        choices=['fer2013', 'ckplus', 'affectnet', 'combined'],
                        help='Which model to use for detection')
    args = parser.parse_args()
    
    # Set the selected model based on command-line argument
    SELECTED_MODEL = args.model
    
    print(f"Starting Emotion Detection Webcam Demo with {SELECTED_MODEL} model")
    print(f"Detected emotion classes: {config.EMOTION_CLASSES}")
    
    # Run webcam detection with a single model
    run_single_model_detection(SELECTED_MODEL)