print("Testing imports...")

try:
    import config
    print("Config imported successfully")
    
    from data_preprocessing.preprocess import load_fer2013
    print("Preprocess module imported successfully")
    
    from models.cnn_model import create_emotion_model
    print("Model module imported successfully")
    
    print("All imports successful!")
except Exception as e:
    print(f"Error during imports: {e}")
    import traceback
    traceback.print_exc() 