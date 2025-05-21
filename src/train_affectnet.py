import os
import logging
import tensorflow as tf
from utils.utils import AFFECTNET_DIR, get_model_dirs
from architecture.cnn_architecture_improved import create_improved_emotion_model
from preprocessing.affectnet_processor import process_affectnet
from visualization.training_plots import create_all_visualizations

# Get model directories
_, MODEL_DIR, VIS_DIR = get_model_dirs('affectnet', 'cnn')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_affectnet.log')
    ]
)

class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.learning_rates.append(lr)

def train_on_affectnet():
    try:
        dataset_folder = AFFECTNET_DIR
        sub_folders = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
        class_names = sub_folders

        logging.info("Processing AffectNet dataset...")
        X_train, X_test, y_train, y_test = process_affectnet(dataset_folder, sub_folders)
        
        logging.info(f"Training data shape: {X_train.shape}")
        logging.info(f"Test data shape: {X_test.shape}")
        logging.info(f"Training labels shape: {y_train.shape}")
        logging.info(f"Test labels shape: {y_test.shape}")
        
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("No data loaded. Check if the dataset directory structure is correct.")

        logging.info("Creating model...")
        model, callbacks = create_improved_emotion_model("affectnet", learning_rate=0.001)

        epochs = 50
        batch_size = 32

        # Learning rate schedule
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # Learning rate logger
        lr_logger = LearningRateLogger()

        logging.info("Starting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[lr_schedule, lr_logger],
            verbose=1
        )

        # Add learning rates to history for plotting
        history.history['lr'] = lr_logger.learning_rates

        logging.info("Evaluating model...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"Test accuracy: {test_accuracy*100:.2f}%")
        logging.info(f"Test loss: {test_loss:.4f}")

        logging.info("Generating visualizations...")
        create_all_visualizations(model, history, X_test, y_test, class_names, "affectnet")

        logging.info("Saving final model...")
        if not os.path.exists('models'):
            os.makedirs('models')
        model.save('models/affectnet_model.h5')
        logging.info("Training completed. Check 'results' directory for visualizations.")

    except Exception as e:
        logging.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        train_on_affectnet()
    except Exception as e:
        logging.error("Fatal error in main", exc_info=True)
        exit(1) 