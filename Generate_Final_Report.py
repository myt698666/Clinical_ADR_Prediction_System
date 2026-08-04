import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the output directory exists
os.makedirs('ADR_Summary', exist_ok=True)

print("Loading model performance scorecards for comparative analysis.")

# Load experimental results
xgb_df = pd.read_csv("ADR_Summary/Final_27_SOC_CV_Results.csv")
mlp_df = pd.read_csv("ADR_Summary/MLP_All_Targets_Results.csv")

# Merge dataframes on System Organ Class (SOC) targets
merged_df = pd.merge(xgb_df, mlp_df, on='SOC_Target', how='inner')

# Compute performance delta between architectures
merged_df['F1_Difference'] = merged_df['MLP_F1_Mean'] - merged_df['F1_Score_Mean']

# Sort by XGBoost performance to establish baseline ranking
merged_df = merged_df.sort_values(by='F1_Score_Mean', ascending=False)

# Export consolidated performance matrix
output_csv = "ADR_Summary/Ultimate_Model_Comparison.csv"
merged_df.to_csv(output_csv, index=False)
print(f"Consolidated performance matrix exported to: {output_csv}")

# Visualization setup
targets = merged_df['SOC_Target'].tolist()
xgb_f1 = merged_df['F1_Score_Mean'].tolist()
mlp_f1 = merged_df['MLP_F1_Mean'].tolist()

x = np.arange(len(targets))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))

# Plot bar charts with academic color schemes
rects1 = ax.bar(x - width/2, xgb_f1, width, label='XGBoost (Gradient Trees)', color='#2c3e50', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x + width/2, mlp_f1, width, label='MLP (Deep Neural Network)', color='#e67e22', edgecolor='black', alpha=0.9)

# Formatting axes and labels
ax.set_ylabel('Mean F1-Score (5-Fold CV)', fontsize=12, fontweight='bold')
ax.set_xlabel('MedDRA System Organ Class (SOC)', fontsize=12, fontweight='bold')
ax.set_title('Performance Benchmarking: Gradient Boosting vs. Deep Learning Architectures', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=11)
ax.legend(fontsize=12, frameon=True)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Highlight targets of interest (Neoplasms and Reproductive disorders)
for i, target in enumerate(targets):
    if target in ['Neopl', 'Repro']:
        ax.get_xticklabels()[i].set_color("#c0392b")
        ax.get_xticklabels()[i].set_fontweight("bold")

fig.tight_layout()

# Export figure with high-resolution settings for publication
output_img = "ADR_Summary/Model_Comparison_BarChart.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Publication-ready comparison chart saved to: {output_img}")
plt.show()