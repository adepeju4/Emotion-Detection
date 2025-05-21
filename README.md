# Emotion Detection Project

This project implements and compares multiple deep learning models for facial emotion recognition using three popular datasets: FER2013, CK+, and AffectNet.

## Overview

The project trains three different CNN models:
1. A model trained on FER2013 dataset
2. A model trained on CK+ dataset
3. A model trained on AffectNet dataset

Each model is evaluated on all datasets to compare performance and generalization capabilities.


## Requirements

- Python 3.9+
- TensorFlow 2.x
- NumPy
- Matplotlib
- OpenCV
- scikit-learn
- pandas
- seaborn



## Dataset Preparation

The project expects datasets to be organized in the following structure:

### FER2013

```
data/Fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

### CK+

```
data/CK+/
├── anger/
├── contempt/
├── disgust/
├── fear/
├── happy/
├── sadness/
└── surprise/
```

### AffectNet

```
data/Affectnet/
├── anger/
├── contempt/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
├── surprise/
└── labels.csv
```
These datasets can be downloaded at the sources listed at the end of this Readme file.


## Usage

### Running the Entire Project

To train all models and evaluate them:

```bash
python main.py
```

### Training Individual Models

To train each model separately:

```bash
python src/train_fer2013.py
python src/train_ckplus.py
python src/train_affectnet.py
```



## Results

The project generates:
- Training history plots for each model
- Confusion matrices for each model on each dataset
- Sample prediction visualizations
- Comprehensive performance metrics including accuracy, precision, recall, and F1-score

## Model Architecture

The project implements two CNN architectures:
1. A standard CNN for larger datasets (FER2013, AffectNet)
2. A lightweight CNN for smaller datasets (CK+)

The standard CNN architecture includes:
- Multiple convolutional blocks with batch normalization
- Max pooling layers
- Dropout for regularization
- Dense layers for classification

## License

[MIT License](LICENSE)

## Acknowledgments
- Basic CNN architecture inspired by [Skillcate](https://medium.com/@skillcate/emotion-detection-model-using-cnn-a-complete-guide-831db1421fae)
- FER2013 dataset: [Kaggle](https://www.kaggle.com/datasets/astraszab/facial-expression-dataset-image-folders-fer2013)
- CK+ dataset: [Papers With Code](https://www.paperswithcode.com/dataset/ck)
- AffectNet dataset: [Kaggle](https://www.kaggle.com/datasets/mstjebashazida/affectnet?select=archive+%283%29)