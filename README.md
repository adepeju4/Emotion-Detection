# Emotion Detection Project

This project implements and compares multiple deep learning models for facial emotion recognition using three popular datasets: FER2013, CK+, and AffectNet.

## Overview

The project trains four different CNN models:
1. A model trained on FER2013 dataset
2. A model trained on CK+ dataset
3. A model trained on AffectNet dataset
4. A combined model trained on all three datasets

Each model is evaluated on all datasets to compare performance and generalization capabilities.

## Project Structure 

emotion_detection/
├── config.py # Configuration parameters
├── main.py # Main script to run the entire project
├── data_preprocessing/ # Data preprocessing modules
│ ├── init.py
│ ├── preprocess.py # Functions to load and preprocess datasets
│ └── data_generator.py # Data generators for training
├── models/ # Model definitions
│ ├── init.py
│ └── cnn_model.py # CNN architectures
├── train/ # Training scripts
│ ├── init.py
│ ├── train_fer2013.py # Script to train on FER2013
│ ├── train_ckplus.py # Script to train on CK+
│ ├── train_affectnet.py # Script to train on AffectNet
│ └── train_combined.py # Script to train on combined dataset
├── utils/ # Utility functions
│ ├── init.py
│ ├── visualization.py # Visualization tools
│ └── metrics.py # Evaluation metrics
└── datasets/ # Dataset directories
├── Fer2013/
├── CK+/
└── Affectnet/


## Requirements

- Python 3.6+
- TensorFlow 2.x
- NumPy
- Matplotlib
- OpenCV
- scikit-learn
- pandas
- seaborn

You can install the required packages using:

```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn pandas seaborn
```


## Dataset Preparation

The project expects datasets to be organized in the following structure:

### FER2013

datasets/Fer2013/
├── train/
│ ├── angry/
│ ├── disgust/
│ ├── fear/
│ ├── happy/
│ ├── neutral/
│ ├── sad/
│ └── surprise/
└── test/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/



### CK+

datasets/CK+/
├── anger/
├── contempt/
├── disgust/
├── fear/
├── happy/
├── sadness/
└── surprise/


### AffectNet

datasets/Affectnet/
├── anger/
├── contempt/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
├── labels.csv


## Usage

### Running the Entire Project

To train all models and evaluate them:

```bash
python main.py
```

### Training Individual Models

To train each model separately:

```bash
python train/train_fer2013.py
python train/train_ckplus.py
python train/train_affectnet.py
python train/train_combined.py
```



## Results

The project generates:
- Training history plots for each model
- Confusion matrices for each model on each dataset
- Sample prediction visualizations
- Comprehensive performance metrics including accuracy, precision, recall, and F1-score

## Model Architecture

The project implements two CNN architectures:
1. A standard CNN for larger datasets (FER2013, AffectNet, Combined)
2. A lightweight CNN for smaller datasets (CK+)

The standard CNN architecture includes:
- Multiple convolutional blocks with batch normalization
- Max pooling layers
- Dropout for regularization
- Dense layers for classification

## License

[MIT License](LICENSE)

## Acknowledgments

- FER2013 dataset: [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- CK+ dataset: [Lucey et al.](http://www.pitt.edu/~emotion/ck-spread.html)
- AffectNet dataset: [Mollahosseini et al.](http://mohammadmahoor.com/affectnet/)