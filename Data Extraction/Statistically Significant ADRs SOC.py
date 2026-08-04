import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
file_path = 'ADR_Summary/soc_disproportionality_results.csv'
soc_results = pd.read_csv(file_path)

# Step 1: Identify top 5 SOC categories by event_count
top_soc = (
    soc_results.groupby('SOC_ABBREV')['event_count']
    .sum()
    .nlargest(5)
    .index
)

# Step 2: Apply adjustments for the top SOC categories
adjusted_results = soc_results.copy()
adjusted_results.loc[adjusted_results['SOC_ABBREV'].isin(top_soc), 'ROR'] *= 1
adjusted_results.loc[adjusted_results['SOC_ABBREV'].isin(top_soc), 'PRR'] *= 1
adjusted_results.loc[adjusted_results['SOC_ABBREV'].isin(top_soc), 'CI_lower'] *= 1
adjusted_results.loc[adjusted_results['SOC_ABBREV'].isin(top_soc), 'event_count'] *= 1

# Step 3: Determine significance
adjusted_results['Significant'] = (
    (adjusted_results['ROR'] > 2) &
    (adjusted_results['PRR'] > 2) &
    (adjusted_results['CI_lower'] > 1) &
    (adjusted_results['event_count'] > 3)
).astype(int)

# Step 4: Create the matrix
significance_matrix = adjusted_results.pivot_table(
    index='Drug',
    columns='SOC_ABBREV',
    values='Significant',
    fill_value=0
)

# Save the matrix to a file (optional)
output_file = 'ADR_Summary/SOC_significance_matrix_Sider_Comparison.csv'
significance_matrix.to_csv(output_file)

# Display the result
print(significance_matrix)

# Verify the matrix contains expected values (0s and 1s)
#assert significance_matrix.isin([0, 1]).all().all(), "Matrix contains invalid values!"

# Set up the figure
#plt.figure(figsize=(12, 8))

# Create the heatmap
#sns.heatmap(
    #significance_matrix,
    #cmap=["grey", "red"],  # Colors for 0 and 1
    #cbar=False,            # Hide the color bar
    #linewidths=0.5,        # Add grid lines
    #linecolor="white",
#)

# Customize the plot
#plt.title("Significance Heatmap (Drug vs SOC Categories)", fontsize=16)
#plt.xlabel("SOC Categories")
#plt.ylabel("Drugs")

# Show the heatmap
#plt.tight_layout()
#plt.show()
