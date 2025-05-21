import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create results directory if it doesn't exist
os.makedirs('results/comparisons', exist_ok=True)

# Model comparison data
model_data = {
    'Model': ['AffectNet', 'CK+', 'FER2013'],
    'Training Time (minutes)': [28, 2, 54],
    'Test Accuracy (%)': [62.51, 100.00, 58.97],
    'Test Loss': [2.6024, 0.1121, 1.9275]
}

# Create DataFrame
df = pd.DataFrame(model_data)

# Style the DataFrame for better visualization
styled_df = df.style.format({
    'Training Time (minutes)': '{:.0f}',
    'Test Accuracy (%)': '{:.2f}',
    'Test Loss': '{:.4f}'
})

# Save as PNG using matplotlib
plt.figure(figsize=(10, 3))
plt.axis('off')
table = plt.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    colColours=['#f2f2f2']*len(df.columns)
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
plt.savefig('results/comparisons/model_comparison_table.png', 
            bbox_inches='tight', 
            dpi=300,
            pad_inches=0.1)
plt.close()

# Create bar plots for accuracy and training time
plt.figure(figsize=(12, 5))

# Accuracy comparison
plt.subplot(1, 2, 1)
sns.barplot(data=df, x='Model', y='Test Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)
plt.ylim(0, 100)

# Training time comparison
plt.subplot(1, 2, 2)
sns.barplot(data=df, x='Model', y='Training Time (minutes)')
plt.title('Training Time Comparison')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('results/comparisons/model_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create loss comparison plot
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Model', y='Test Loss')
plt.title('Model Loss Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/comparisons/model_loss_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Comparison visualizations have been saved to results/comparisons/") 