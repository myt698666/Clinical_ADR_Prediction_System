import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Load input and output files
input_file_path = "STITCH_Identifiers/drug_protein_interaction_matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

# Merge and clean data
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')
merged_data = merged_data.dropna()

# Select 10% of rows randomly
subset_rows = merged_data.sample(frac=0.05, random_state=42).sort_values(by="Drug")

# Select 5% of input columns
input_cols = inputs.columns
num_input_cols = max(1, int(0.02 * len(input_cols)))  # Ensure at least one column is selected
selected_input_cols = sorted(np.random.choice(input_cols, num_input_cols, replace=False))

# Keep all output columns
output_cols = outputs.columns

# Create final subset
final_subset = subset_rows[list(selected_input_cols) + list(output_cols)]

# Define custom colormap
custom_cmap = LinearSegmentedColormap.from_list(
    "custom", ["#F1F2F3", "#12616E"], N=256
)

# Generate heatmap
plt.figure(figsize=(18, 8),facecolor='none')
ax=sns.heatmap(final_subset.set_index("Drug"), cmap=custom_cmap, annot=False, cbar=False,square=True)
# Adjust label density
ax.set_xticks(ax.get_xticks()[::2])  # Show every 2nd tick on x-axis
ax.set_yticks(ax.get_yticks()[::3])  # Show every 3rd tick on y-axis

# Rotate and resize labels for better readability
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)

plt.title("Heatmap of Selected Drug-Protein and ADR Interactions")
plt.show()
