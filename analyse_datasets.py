#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import h5py
import humanize
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def get_file_size(file_path):
    """Get size of a file in bytes."""
    return os.path.getsize(file_path)

def load_dataset(file_path):
    """
    Load dataset based on file extension
    
    Returns:
        data: Dataset as pandas DataFrame or dict
        type: Type of dataset
    """
    file_path = str(file_path)
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path), 'tabular'
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path), 'tabular'
    elif file_path.endswith('.json'):
        with open(file_path) as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return pd.DataFrame(data), 'tabular'
        return data, 'json'
    elif file_path.endswith(('.h5', '.hdf5')):
        with h5py.File(file_path, 'r') as f:
            # Extract first dataset found
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    return np.array(f[key]), 'array'
            return None, 'unknown'
    elif file_path.endswith('.npy'):
        return np.load(file_path), 'array'
    elif file_path.endswith('.npz'):
        data = np.load(file_path)
        # Return the first array in the file
        for key in data.keys():
            return data[key], 'array'
    else:
        return None, 'unknown'

def analyze_tabular_data(df):
    """Analyze a tabular dataset (pandas DataFrame)"""
    results = {
        'rows': len(df),
        'columns': len(df.columns),
        'column_types': {},
        'numeric_stats': {},
        'categorical_stats': {},
        'missing_values': {}
    }
    
    for col in df.columns:
        # Get data type
        dtype = df[col].dtype
        results['column_types'][col] = str(dtype)
        
        # Count missing values
        missing = df[col].isna().sum()
        results['missing_values'][col] = missing
        
        # Analyze based on type
        if pd.api.types.is_numeric_dtype(dtype):
            try:
                stats = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std())
                }
                results['numeric_stats'][col] = stats
            except:
                pass
        else:
            # Categorical analysis
            try:
                value_counts = df[col].value_counts().head(5).to_dict()
                unique_count = df[col].nunique()
                results['categorical_stats'][col] = {
                    'unique_values': unique_count,
                    'top_values': value_counts
                }
            except:
                pass
    
    return results

def analyze_array_data(arr):
    """Analyze a numpy array dataset"""
    results = {
        'shape': arr.shape,
        'dtype': str(arr.dtype),
    }
    
    if np.issubdtype(arr.dtype, np.number):
        results.update({
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr))
        })
    
    return results

def scan_directory(directory, extensions=None, max_rows=1000):
    """
    Scan directory for dataset files and analyze their contents
    
    Args:
        directory (str): Directory path to scan
        extensions (list): File extensions to include
        max_rows (int): Maximum number of rows to analyze for large datasets
    
    Returns:
        dict: Dictionary with dataset paths and their analysis
    """
    dataset_analysis = {}
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return dataset_analysis
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            
            # Skip if we're filtering by extension and this file doesn't match
            if extensions and not any(file.endswith(ext) for ext in extensions):
                continue
                
            try:
                size = get_file_size(file_path)
                
                print(f"Analyzing {file_path}...")
                try:
                    data, data_type = load_dataset(file_path)
                    
                    analysis = {
                        'file_size': size,
                        'file_size_human': humanize.naturalsize(size),
                        'data_type': data_type,
                    }
                    
                    if data_type == 'tabular':
                        if len(data) > max_rows:
                            data = data.sample(max_rows)
                        analysis.update(analyze_tabular_data(data))
                    elif data_type == 'array':
                        analysis.update(analyze_array_data(data))
                    
                    dataset_analysis[str(file_path)] = analysis
                except Exception as e:
                    print(f"  Error analyzing data: {e}")
                    dataset_analysis[str(file_path)] = {
                        'file_size': size,
                        'file_size_human': humanize.naturalsize(size),
                        'error': str(e)
                    }
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return dataset_analysis

def display_tabular_summary(analysis_results):
    """Display a summary table of the tabular datasets"""
    table_data = []
    
    for path, analysis in analysis_results.items():
        if analysis.get('data_type') == 'tabular':
            row = [
                os.path.basename(path),
                analysis.get('file_size_human', 'N/A'),
                analysis.get('rows', 'N/A'),
                analysis.get('columns', 'N/A'),
                sum(analysis.get('missing_values', {}).values()),
                len(analysis.get('numeric_stats', {})),
                len(analysis.get('categorical_stats', {}))
            ]
            table_data.append(row)
    
    headers = ["Dataset", "Size", "Rows", "Columns", "Missing Values", 
               "Numeric Cols", "Categorical Cols"]
    
    print("\nTABULAR DATASET SUMMARY")
    print("======================")
    print(tabulate(table_data, headers, tablefmt="grid"))

def display_column_distributions(analysis_results):
    """Display distributions of numeric columns"""
    numeric_stats = defaultdict(list)
    
    for path, analysis in analysis_results.items():
        if analysis.get('data_type') == 'tabular':
            dataset_name = os.path.basename(path)
            for col, stats in analysis.get('numeric_stats', {}).items():
                row = [
                    dataset_name,
                    col,
                    stats.get('min', 'N/A'),
                    stats.get('max', 'N/A'),
                    stats.get('mean', 'N/A'),
                    stats.get('median', 'N/A'),
                    stats.get('std', 'N/A')
                ]
                numeric_stats['data'].append(row)
    
    if numeric_stats['data']:
        headers = ["Dataset", "Column", "Min", "Max", "Mean", "Median", "Std Dev"]
        
        print("\nNUMERIC COLUMN DISTRIBUTIONS")
        print("===========================")
        print(tabulate(numeric_stats['data'], headers, tablefmt="grid"))

def display_array_summary(analysis_results):
    """Display summary of array datasets"""
    table_data = []
    
    for path, analysis in analysis_results.items():
        if analysis.get('data_type') == 'array':
            row = [
                os.path.basename(path),
                analysis.get('file_size_human', 'N/A'),
                str(analysis.get('shape', 'N/A')),
                analysis.get('dtype', 'N/A')
            ]
            
            if 'mean' in analysis:
                row.extend([
                    analysis.get('min', 'N/A'),
                    analysis.get('max', 'N/A'),
                    analysis.get('mean', 'N/A'),
                    analysis.get('std', 'N/A')
                ])
            else:
                row.extend(['N/A', 'N/A', 'N/A', 'N/A'])
            
            table_data.append(row)
    
    if table_data:
        headers = ["Dataset", "Size", "Shape", "Data Type", 
                   "Min", "Max", "Mean", "Std Dev"]
        
        print("\nARRAY DATASET SUMMARY")
        print("====================")
        print(tabulate(table_data, headers, tablefmt="grid"))

def analyze_emotion_distribution(dataset_dir):
    """Analyze the distribution of emotions in the dataset"""
    emotion_counts = {}
    
    # Check if this is a directory-based dataset
    if os.path.isdir(dataset_dir):
        # First check if this is a dataset with train/test splits
        split_dirs = ['train', 'test', 'val', 'validation']
        has_splits = any(os.path.isdir(os.path.join(dataset_dir, d)) for d in split_dirs)
        
        if has_splits:
            # Process each split directory
            for split in split_dirs:
                split_path = os.path.join(dataset_dir, split)
                if os.path.isdir(split_path):
                    # Get emotion directories within this split
                    emotion_dirs = [d for d in os.listdir(split_path) 
                                   if os.path.isdir(os.path.join(split_path, d))]
                    
                    for emotion in emotion_dirs:
                        emotion_path = os.path.join(split_path, emotion)
                        # Count images in this emotion directory
                        image_count = len([f for f in os.listdir(emotion_path) 
                                         if os.path.isfile(os.path.join(emotion_path, f)) and 
                                         f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        
                        # Use the original emotion name as the key
                        emotion_key = emotion
                        emotion_counts[emotion_key] = emotion_counts.get(emotion_key, 0) + image_count
        else:
            # Direct emotion directories (like in your AffectNet and CK+)
            emotion_dirs = [d for d in os.listdir(dataset_dir) 
                           if os.path.isdir(os.path.join(dataset_dir, d))]
            
            for emotion in emotion_dirs:
                emotion_path = os.path.join(dataset_dir, emotion)
                # Count images in this emotion directory
                image_count = len([f for f in os.listdir(emotion_path) 
                                  if os.path.isfile(os.path.join(emotion_path, f)) and 
                                  f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                # Use the original emotion name
                emotion_counts[emotion] = image_count
    
    # CSV file handling remains the same
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    for csv_file in csv_files:
        csv_path = os.path.join(dataset_dir, csv_file)
        try:
            df = pd.read_csv(csv_path)
            # Look for emotion column (common names)
            emotion_col = None
            for col in ['emotion', 'label', 'class', 'expression']:
                if col in df.columns:
                    emotion_col = col
                    break
            
            if emotion_col:
                # Count emotions
                emotion_distribution = df[emotion_col].value_counts().to_dict()
                # Try to map numeric labels to emotion names if needed
                if all(isinstance(k, (int, np.integer)) for k in emotion_distribution.keys()):
                    # For FER2013 dataset
                    fer_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                    emotion_counts = {fer_emotions[k]: v for k, v in emotion_distribution.items() 
                                     if k < len(fer_emotions)}
                else:
                    emotion_counts.update(emotion_distribution)
        except Exception as e:
            print(f"Error analyzing CSV file {csv_file}: {e}")
    
    return emotion_counts

def main():
    parser = argparse.ArgumentParser(description='Analyze datasets and their distributions')
    parser.add_argument('--dirs', nargs='+', required=True, 
                        help='Directories to scan for datasets')
    parser.add_argument('--extensions', nargs='+', 
                        default=['.csv', '.json', '.parquet', '.h5', '.hdf5', '.npy', '.npz'],
                        help='File extensions to include (default: common dataset formats)')
    parser.add_argument('--max-rows', type=int, default=10000,
                        help='Maximum number of rows to analyze for large datasets')
    
    args = parser.parse_args()
    
    all_analysis = {}
    for directory in args.dirs:
        results = scan_directory(directory, args.extensions, args.max_rows)
        all_analysis.update(results)
    
    # Display results in tables
    display_tabular_summary(all_analysis)
    display_column_distributions(all_analysis)
    display_array_summary(all_analysis)
    
    # Summary
    print("\nSUMMARY")
    print("=======")
    total_size = sum(analysis.get('file_size', 0) for analysis in all_analysis.values())
    print(f"Total datasets analyzed: {len(all_analysis)}")
    print(f"Total size: {humanize.naturalsize(total_size)}")
    
    # Emotion distribution
    print("\nEMOTION DISTRIBUTION")
    print("====================")
    emotion_table = []
    for directory in args.dirs:
        if os.path.exists(directory):
            dataset_name = os.path.basename(directory)
            emotion_counts = analyze_emotion_distribution(directory)
            if emotion_counts:
                for emotion, count in emotion_counts.items():
                    emotion_table.append([dataset_name, emotion, count])

    if emotion_table:
        emotion_df = pd.DataFrame(emotion_table, columns=["Dataset", "Emotion", "Count"])
        
        # Create a pivot table with datasets as rows and emotions as columns
        pivot_df = emotion_df.pivot_table(index="Dataset", columns="Emotion", values="Count", fill_value=0)
        
        # Print the table
        print(tabulate(pivot_df, headers="keys", tablefmt="grid"))
        
        # Add percentage distribution for each dataset separately
        print("\nEMOTION DISTRIBUTION (PERCENTAGE)")
        print("================================")
        for dataset in pivot_df.index:
            # Get only the columns that have non-zero values for this dataset
            dataset_emotions = pivot_df.loc[dataset]
            dataset_emotions = dataset_emotions[dataset_emotions > 0]
            
            # Calculate percentages
            total = dataset_emotions.sum()
            percentage_df = (dataset_emotions / total * 100).round(1)
            
            print(f"\n{dataset}:")
            print(tabulate(percentage_df.to_frame().T, headers="keys", tablefmt="grid"))
        
        # Create a summary of emotion naming differences
        print("\nEMOTION NAMING DIFFERENCES")
        print("=========================")
        print("Note: Different datasets use different names for the same emotions:")
        print("- FER2013: 'angry', 'sad'")
        print("- CK+: 'anger', 'sadness'")
        print("- AffectNet: 'anger', 'sad'")
        print("\nThis explains why some emotions appear to have zero counts in certain datasets.")
    else:
        print("No emotion distribution data found.")
    
if __name__ == "__main__":
    main()