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

# Pivoting the data
heatmap_data_event_count = soc_results.pivot(index='SOC_ABBREV', columns='Drug', values='event_count')

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
    index='SOC_ABBREV',
    columns='Drug',
    values='Significant',
    fill_value=0
)

# Plotting the heatmaps side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Heatmap for event_count
sns.heatmap(
    heatmap_data_event_count, ax=axes[0], cmap='YlGnBu',
    cbar_kws={'label': 'Event Count'}, robust=True
)
axes[0].set_title('Event Count Heatmap')
axes[0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# Heatmap for significance
sns.heatmap(
    significance_matrix, ax=axes[1],
    cmap=["white", "red"],  # Colors for 0 and 1
    cbar=True,            # Display the color bar
    linecolor="white"
)
axes[1].set_title("Significance Heatmap (SOC Categories vs Drugs)")
axes[1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# Show the heatmap
plt.tight_layout()
plt.show()
