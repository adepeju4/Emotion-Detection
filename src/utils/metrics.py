from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

def calculate_metrics(y_true, y_pred):
    """Calculate various metrics for model evaluation."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics 