# webcam_emotion_detection.py
import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
    'neutral': (255, 255, 255)  # White
}

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
        scaleFactor=1.1,       # Smaller scale factor for better detection
        minNeighbors=5,        # Higher threshold to reduce false positives
        minSize=(min_face_size, min_face_size),  # Minimum face size
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
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
    """Run real-time emotion detection on webcam feed"""
    # Check if model exists, if not, inform user
    if not os.path.exists(config.CKPLUS_MODEL_PATH):
        print(f"Model not found at {config.CKPLUS_MODEL_PATH}")
        print("Please train the model first or specify the correct model path.")
        return
    
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(config.CKPLUS_MODEL_PATH)
    
    # Print model output shape for debugging
    print(f"Model output shape: {model.output_shape}")
    num_classes = model.output_shape[-1]
    print(f"Number of output classes: {num_classes}")
    
    # Determine which emotion set to use based on the model's output shape
    if num_classes == len(config.FER2013_EMOTIONS):
        emotion_classes = config.FER2013_EMOTIONS
        print("Using FER2013 emotion classes")
    elif num_classes == len(config.CKPLUS_EMOTIONS):
        emotion_classes = config.CKPLUS_EMOTIONS
        print("Using CK+ emotion classes")
    elif num_classes == len(config.AFFECTNET_EMOTIONS):
        emotion_classes = config.AFFECTNET_EMOTIONS
        print("Using AffectNet emotion classes")
    elif num_classes == len(config.COMBINED_MODEL_EMOTIONS):
        emotion_classes = config.COMBINED_MODEL_EMOTIONS
        print("Using combined emotion classes")
    else:
        # Fallback to default
        emotion_classes = config.COMMON_EMOTIONS
        print(f"Using common emotion classes (model has {num_classes} outputs)")
        if num_classes != len(emotion_classes):
            print("WARNING: Model output classes don't match emotion classes!")
    
    print(f"Emotion classes for this model: {emotion_classes}")
    
    # Start webcam
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get webcam dimensions
    webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create emotion bar chart
    sidebar_width = 400  # Width of the sidebar in pixels
    chart_height = webcam_height // 2  # Height of the chart (half of webcam height)
    
    fig, ax, bars, _ = create_emotion_bar_chart(
        figsize=(sidebar_width/100, chart_height/100),
        emotion_classes=emotion_classes
    )
    chart_image = None
    
    # Create a combined display frame (webcam feed + sidebar)
    combined_width = webcam_width + sidebar_width
    combined_height = webcam_height
    
    print("Webcam started. Press 'q' to quit.")
    
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
        
        # Preprocess frame
        faces, face_locations = preprocess_frame(frame)
        
        # Create combined frame (black background)
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Add padding to sidebar (dark gray background)
        sidebar_padding = 15  # Increased padding
        cv2.rectangle(combined_frame, 
                     (webcam_width + sidebar_padding, sidebar_padding), 
                     (combined_width - sidebar_padding, combined_height - sidebar_padding), 
                     (50, 50, 50), -1)  # Dark gray background
        
        # Place webcam feed on the left side
        combined_frame[0:webcam_height, 0:webcam_width] = display_frame
        
        if faces is not None and len(faces) > 0:
            # Make predictions for each face
            predictions = model.predict(faces)
            
            # Update bar chart with prediction from the first face
            chart_image = update_emotion_bar_chart(fig, ax, bars, predictions[0])
            
            # Display results for each face
            for i, (x, y, w, h) in enumerate(face_locations):
                # Get prediction for this face
                prediction = predictions[i]
                emotion_idx = np.argmax(prediction)
                emotion = emotion_classes[emotion_idx]
                confidence = prediction[emotion_idx] * 100
                
                # Get color for this emotion
                color = EMOTION_COLORS.get(emotion, (0, 255, 0))
                
                # Draw rectangle around face (1px)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 1)
                
                # Create background for text
                text = f"{emotion} ({confidence:.1f}%)"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
                cv2.rectangle(display_frame, 
                              (x, y - text_size[1] - 10), 
                              (x + text_size[0], y), 
                              color, -1)  # Filled rectangle
                
                # Display emotion text (white on colored background)
                cv2.putText(display_frame, text, (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Update the webcam feed in the combined frame
            combined_frame[0:webcam_height, 0:webcam_width] = display_frame
        
        # Calculate and display FPS
        frame_count += 1
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
        
        # Add emotion bar chart to the bottom of the sidebar if available
        if chart_image is not None:
            # Calculate position for chart (bottom of sidebar with padding)
            chart_y = combined_height - chart_height - sidebar_padding * 2
            
            # Resize chart to fit in the sidebar with padding
            resized_chart = cv2.resize(chart_image, 
                                      (sidebar_width - sidebar_padding * 2, chart_height))
            
            # Place the chart in the sidebar
            combined_frame[chart_y:chart_y+chart_height, 
                          webcam_width+sidebar_padding:webcam_width+sidebar_width-sidebar_padding] = resized_chart
            
            # Add title above the chart
            cv2.putText(combined_frame, "Detected Emotions", 
                       (webcam_width + sidebar_padding + 10, chart_y - 20), 
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

if __name__ == "__main__":
    print("Starting Emotion Detection Webcam Demo")
    print(f"Detected emotion classes: {config.EMOTION_CLASSES}")
    run_webcam_detection()