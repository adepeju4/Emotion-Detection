import os
import logging
import tensorflow as tf
from datetime import datetime


from src.train_fer2013 import train_fer2013
from src.train_ckplus import train_on_ckplus
from src.train_affectnet import train_on_affectnet


from src.utils.utils import FER2013_DIR, CKPLUS_DIR, AFFECTNET_DIR
from src.visualization.generate_comparison_tables import generate_comparison_tables
from src.visualization.generate_model_visualizations import generate_model_visualizations

# Configure logging
def setup_logging():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'training_main_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename)
        ]
    )
    return logging.getLogger(__name__)

def verify_datasets():
    datasets = {
        'FER2013': FER2013_DIR,
        'CK+': CKPLUS_DIR,
        'AffectNet': AFFECTNET_DIR
    }
    
    missing_datasets = []
    for name, path in datasets.items():
        if not os.path.exists(path):
            missing_datasets.append(name)
    
    return missing_datasets

def create_required_directories():
    required_dirs = ['models', 'results']
    for dir_name in required_dirs:
        os.makedirs(dir_name, exist_ok=True)

def train_all_models(logger):
    models_status = {
        'FER2013': False,
        'CK+': False,
        'AffectNet': False
    }
    
    try:
        logger.info("Starting FER2013 model training...")
        train_fer2013()
        models_status['FER2013'] = True
        logger.info("FER2013 model training completed successfully")
    except Exception as e:
        logger.error(f"Error training FER2013 model: {str(e)}", exc_info=True)

    try:
        logger.info("Starting CK+ model training...")
        train_on_ckplus()
        models_status['CK+'] = True
        logger.info("CK+ model training completed successfully")
    except Exception as e:
        logger.error(f"Error training CK+ model: {str(e)}", exc_info=True)

    try:
        logger.info("Starting AffectNet model training...")
        train_on_affectnet()
        models_status['AffectNet'] = True
        logger.info("AffectNet model training completed successfully")
    except Exception as e:
        logger.error(f"Error training AffectNet model: {str(e)}", exc_info=True)
    
    return models_status

def evaluate_models(logger):
    try:
        models = {
            'fer2013': tf.keras.models.load_model('models/fer2013_model.h5'),
            'ckplus': tf.keras.models.load_model('models/ckplus_model.h5'),
            'affectnet': tf.keras.models.load_model('models/affectnet_model.h5')
        }
        

        logger.info("Generating model comparison tables...")
        generate_comparison_tables(models)
        
        logger.info("Generating model visualizations...")
        generate_model_visualizations(models)
        
        logger.info("Model evaluation and comparison completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}", exc_info=True)
        return False

def main():
    logger = setup_logging()
    logger.info("Starting emotion detection model training and evaluation pipeline")
    
    create_required_directories()
    
    missing_datasets = verify_datasets()
    if missing_datasets:
        logger.error(f"Missing datasets: {', '.join(missing_datasets)}")
        logger.error("Please ensure all datasets are properly set up according to README.md")
        return
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), enabled memory growth")
    except Exception as e:
        logger.warning(f"Error configuring GPU: {str(e)}")
    
    logger.info("Starting training pipeline for all models")
    models_status = train_all_models(logger)
    
    if not any(models_status.values()):
        logger.error("No models were trained successfully. Stopping evaluation.")
        return
    
    logger.info("Starting model evaluation and comparison")
    evaluation_success = evaluate_models(logger)
    
    logger.info("\n=== Final Status Report ===")
    for model, status in models_status.items():
        logger.info(f"{model} Model: {'Successfully trained' if status else 'Failed'}")
    logger.info(f"Model Evaluation: {'Completed' if evaluation_success else 'Failed'}")
    logger.info("Pipeline execution completed")

if __name__ == "__main__":
    main()
