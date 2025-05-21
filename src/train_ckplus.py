import os
import logging
import tensorflow as tf
from utils.utils import CKPLUS_DIR
from architecture.cnn_architecture import create_emotion_model
from preprocessing.ck_plus_processor import process_ck_plus
from visualization.training_plots import create_all_visualizations


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_ckplus.log')
    ]
)

class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.learning_rates.append(lr)

def train_on_ckplus():
    try:
        dataset_folder = CKPLUS_DIR
        sub_folders = ["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"]
        class_names = sub_folders

        logging.info("Processing CK+ dataset...")
        X_train, X_test, y_train, y_test = process_ck_plus(dataset_folder, sub_folders)

        logging.info("Creating model...")
        model = create_emotion_model("ckplus")

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
        create_all_visualizations(model, history, X_test, y_test, class_names, "ckplus")

        logging.info("Saving final model...")
        if not os.path.exists('models'):
            os.makedirs('models')
        model.save('models/ckplus_model.h5')
        logging.info("Training completed. Check 'results' directory for visualizations.")

    except Exception as e:
        logging.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        train_on_ckplus()
    except Exception as e:
        logging.error("Fatal error in main", exc_info=True)
        exit(1)
