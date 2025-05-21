import numpy as np
import matplotlib.pyplot as plt

# Dataset statistics
datasets = {
    'FER2013': {
        'anger': 4953,
        'disgust': 547,
        'fear': 5121,
        'happiness': 8989,
        'neutral': 6198,
        'sadness': 6077,
        'surprise': 4002,
        'contempt': 0  # Not present in FER2013
    },
    'CK+': {
        'anger': 135,
        'contempt': 54,
        'disgust': 177,
        'fear': 75,
        'happiness': 207,
        'sadness': 84,
        'surprise': 249,
        'neutral': 0  # Not present in CK+
    },
    'AffectNet': {
        'anger': 3608,
        'contempt': 3244,
        'disgust': 3472,
        'fear': 3043,
        'happiness': 4336,
        'neutral': 2861,
        'sadness': 2995,
        'surprise': 4616
    }
}

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

emotions = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
dataset_names = list(datasets.keys())

# Create cell text with totals column
cell_text = []
for dataset in dataset_names:
    row = [f"{datasets[dataset][emotion]:,}" for emotion in emotions]
    total = sum(datasets[dataset].values())
    row.append(f"{total:,}")  # Add total for this dataset
    cell_text.append(row)

# Add column labels including Total
col_labels = emotions + ['Total']

table = ax.table(cellText=cell_text,
                rowLabels=dataset_names,
                colLabels=col_labels,
                loc='center',
                cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)

# Style column headers
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white')

# Style row headers and total column
for i in range(len(dataset_names)):
    # Style row headers
    table[i + 1, -1].set_facecolor('#4472C4')
    table[i + 1, -1].set_text_props(color='white')
    # Style total column
    table[i + 1, len(emotions)].set_facecolor('#E7E6E6')

plt.title('Emotion Distribution Across Datasets', pad=20, size=14)

plt.savefig('results/datasets_analysis/dataset_comparison.png', bbox_inches='tight', dpi=300)
plt.close() 