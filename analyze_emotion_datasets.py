#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import json

def count_images_in_directory(directory):
    """Count image files in a directory"""
    return len([f for f in os.listdir(directory) 
               if os.path.isfile(os.path.join(directory, f)) and 
               f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])

def analyze_fer2013(dataset_dir):
    """Analyze FER2013 dataset"""
    emotion_counts = {}
    
    # FER2013 has train and test directories
    for split in ['train', 'test']:
        split_path = os.path.join(dataset_dir, split)
        if os.path.isdir(split_path):
            # Get emotion directories
            emotion_dirs = [d for d in os.listdir(split_path) 
                           if os.path.isdir(os.path.join(split_path, d))]
            
            for emotion in emotion_dirs:
                emotion_path = os.path.join(split_path, emotion)
                image_count = count_images_in_directory(emotion_path)
                
                # Add to existing count or create new entry
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + image_count
    
    return emotion_counts

def analyze_ckplus(dataset_dir):
    """Analyze CK+ dataset"""
    emotion_counts = {}
    
    # CK+ has emotion directories directly
    emotion_dirs = [d for d in os.listdir(dataset_dir) 
                   if os.path.isdir(os.path.join(dataset_dir, d))]
    
    for emotion in emotion_dirs:
        emotion_path = os.path.join(dataset_dir, emotion)
        image_count = count_images_in_directory(emotion_path)
        emotion_counts[emotion] = image_count
    
    return emotion_counts

def analyze_affectnet(dataset_dir):
    """Analyze AffectNet dataset"""
    emotion_counts = {}
    
    # AffectNet has emotion directories directly
    emotion_dirs = [d for d in os.listdir(dataset_dir) 
                   if os.path.isdir(os.path.join(dataset_dir, d))]
    
    for emotion in emotion_dirs:
        emotion_path = os.path.join(dataset_dir, emotion)
        image_count = count_images_in_directory(emotion_path)
        emotion_counts[emotion] = image_count
    
    return emotion_counts

def create_emotion_distribution_table(datasets):
    """Create a table of emotion distributions"""
    all_data = []
    
    for dataset_name, counts in datasets.items():
        for emotion, count in counts.items():
            all_data.append([dataset_name, emotion, count])
    
    # Create DataFrame and pivot
    df = pd.DataFrame(all_data, columns=["Dataset", "Emotion", "Count"])
    pivot_df = df.pivot_table(index="Dataset", columns="Emotion", values="Count", fill_value=0)
    
    return pivot_df

def plot_emotion_distribution(pivot_df, output_file=None):
    """Plot emotion distribution as a bar chart"""
    plt.figure(figsize=(14, 8))
    
    # Plot each dataset as a group of bars
    bar_width = 0.25
    datasets = pivot_df.index
    emotions = pivot_df.columns
    x = np.arange(len(emotions))
    
    for i, dataset in enumerate(datasets):
        offset = (i - len(datasets)/2 + 0.5) * bar_width
        plt.bar(x + offset, pivot_df.loc[dataset], width=bar_width, label=dataset)
    
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Emotion Distribution Across Datasets', fontsize=14)
    plt.xticks(x, emotions, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()

def plot_percentage_distribution(pivot_df, output_file=None):
    """Plot percentage distribution as a stacked bar chart"""
    # Calculate percentages
    percentage_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(14, 8))
    percentage_df.plot(kind='bar', stacked=True, figsize=(14, 8))
    
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title('Emotion Distribution Percentage by Dataset', fontsize=14)
    plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()

def standardize_emotion_labels(datasets):
    """Standardize emotion labels across datasets"""
    # Define mapping from dataset-specific labels to standard labels
    mappings = {
        'FER2013': {
            'angry': 'anger',
            'sad': 'sadness',
            'happy': 'happiness',
            'disgust': 'disgust',
            'fear': 'fear',
            'surprise': 'surprise',
            'neutral': 'neutral'
        },
        'CK+': {
            'anger': 'anger',
            'contempt': 'contempt',
            'disgust': 'disgust',
            'fear': 'fear',
            'happy': 'happiness',
            'sadness': 'sadness',
            'surprise': 'surprise'
        },
        'AffectNet': {
            'anger': 'anger',
            'contempt': 'contempt',
            'disgust': 'disgust',
            'fear': 'fear',
            'happy': 'happiness',
            'neutral': 'neutral',
            'sad': 'sadness',
            'surprise': 'surprise'
        }
    }
    
    standardized_datasets = {}
    
    for dataset_name, counts in datasets.items():
        mapping = mappings.get(dataset_name, {})
        standardized_counts = {}
        
        for emotion, count in counts.items():
            # Map to standard label if exists, otherwise keep original
            standard_emotion = mapping.get(emotion, emotion)
            
            # Add to existing count or create new entry
            if standard_emotion in standardized_counts:
                standardized_counts[standard_emotion] += count
            else:
                standardized_counts[standard_emotion] = count
        
        standardized_datasets[dataset_name] = standardized_counts
    
    return standardized_datasets

def generate_emotion_mapping_file(datasets, standardized_datasets, output_file):
    """Generate a JSON file with emotion mappings for each dataset"""
    # Define the standard mapping explicitly
    mappings = {
        'FER2013': {
            'angry': 'anger',
            'sad': 'sadness',
            'happy': 'happiness',
            'disgust': 'disgust',
            'fear': 'fear',
            'surprise': 'surprise',
            'neutral': 'neutral'
        },
        'CK+': {
            'anger': 'anger',
            'contempt': 'contempt',
            'disgust': 'disgust',
            'fear': 'fear',
            'happy': 'happiness',
            'sadness': 'sadness',
            'surprise': 'surprise'
        },
        'AffectNet': {
            'anger': 'anger',
            'contempt': 'contempt',
            'disgust': 'disgust',
            'fear': 'fear',
            'happy': 'happiness',
            'neutral': 'neutral',
            'sad': 'sadness',
            'surprise': 'surprise'
        }
    }
    
    # Add a standard emotion order that can be used across all models
    all_standard_emotions = set()
    for dataset_mapping in mappings.values():
        all_standard_emotions.update(dataset_mapping.values())
    
    # Create a standard order (alphabetical, but with neutral at the end if present)
    standard_order = sorted(all_standard_emotions)
    if 'neutral' in standard_order:
        standard_order.remove('neutral')
        standard_order.append('neutral')
    
    # Create a new dictionary with all mappings
    result_mappings = {
        'standard_order': standard_order
    }
    
    # Copy the original mappings
    for dataset_name, mapping in mappings.items():
        result_mappings[dataset_name] = mapping
    
    # Add reverse mappings for each dataset (standard to dataset-specific)
    for dataset_name, mapping in mappings.items():
        reverse_mapping = {}
        for orig, std in mapping.items():
            if std not in reverse_mapping:
                reverse_mapping[std] = orig
        result_mappings[f"{dataset_name}_reverse"] = reverse_mapping
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(result_mappings, f, indent=4)
    
    print(f"Emotion mappings saved to {output_file}")
    return result_mappings

def main():
    parser = argparse.ArgumentParser(description='Analyze emotion datasets')
    parser.add_argument('--fer2013', type=str, required=True, help='Path to FER2013 dataset')
    parser.add_argument('--ckplus', type=str, required=True, help='Path to CK+ dataset')
    parser.add_argument('--affectnet', type=str, required=True, help='Path to AffectNet dataset')
    parser.add_argument('--output', type=str, help='Output directory for plots')
    parser.add_argument('--verbose', action='store_true', 
                        help='Show detailed analysis and additional information')
    
    args = parser.parse_args()
    
    # Analyze datasets
    if args.verbose:
        print("Analyzing FER2013 dataset...")
    fer2013_counts = analyze_fer2013(args.fer2013)
    
    if args.verbose:
        print("Analyzing CK+ dataset...")
    ckplus_counts = analyze_ckplus(args.ckplus)
    
    if args.verbose:
        print("Analyzing AffectNet dataset...")
    affectnet_counts = analyze_affectnet(args.affectnet)
    
    # Combine results
    datasets = {
        'FER2013': fer2013_counts,
        'CK+': ckplus_counts,
        'AffectNet': affectnet_counts
    }
    
    # Standardize emotion labels
    if args.verbose:
        print("\nStandardizing emotion labels across datasets...")
    standardized_datasets = standardize_emotion_labels(datasets)
    
    # Create standardized distribution table
    pivot_df = create_emotion_distribution_table(standardized_datasets)
    
    # Display only the standardized counts
    print("\nSTANDARDIZED EMOTION DISTRIBUTION")
    print("================================")
    print(tabulate(pivot_df, headers="keys", tablefmt="grid"))
    
    # Generate emotion mapping file if output directory is specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        mapping_file = os.path.join(args.output, 'emotion_mappings.json')
        mappings = generate_emotion_mapping_file(datasets, standardized_datasets, mapping_file)
        
        if args.verbose:
            # Print the standard emotion order
            print("\nSTANDARD EMOTION ORDER")
            print("=====================")
            print(mappings['standard_order'])
            print("\nUse this order in your config.py and model training scripts for consistency.")
        
        # Create plots
        plot_emotion_distribution(pivot_df, os.path.join(args.output, 'emotion_distribution.png'))
        plot_percentage_distribution(pivot_df, os.path.join(args.output, 'emotion_percentage.png'))
        
        if args.verbose:
            print(f"\nPlots saved to {args.output}")
    elif args.verbose:
        # Create plots only if verbose mode is on
        print("\nGenerating plots...")
        plot_emotion_distribution(pivot_df)
        plot_percentage_distribution(pivot_df)

if __name__ == "__main__":
    main() 