import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with various metrics"""
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Convert one-hot encoded labels to class indices
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate per-class metrics
    cm = confusion_matrix(y_test, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrix': cm
    }
    
    return metrics

def compare_models(models, datasets, class_names=None):
    """Compare performance of multiple models on multiple datasets"""
    results = {}
    
    for model_name, model in models.items():
        model_results = {}
        
        for dataset_name, (X, y) in datasets.items():
            # Evaluate model on dataset
            metrics = evaluate_model(model, X, y)
            model_results[dataset_name] = metrics
        
        results[model_name] = model_results
    
    # Print comparison
    print("Model Comparison:")
    print("=" * 80)
    
    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        print("-" * 40)
        
        for dataset_name, metrics in model_results.items():
            print(f"  Dataset: {dataset_name}")
            print(f"    Accuracy:  {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1 Score:  {metrics['f1_score']:.4f}")
            
            if class_names:
                print("    Per-class accuracy:")
                for i, acc in enumerate(metrics['per_class_accuracy']):
                    print(f"      {class_names[i]}: {acc:.4f}")
    
    return results 