import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the results data from the CSV file
file_path = 'ADR_Summary/soc_disproportionality_results.csv'
soc_results = pd.read_csv(file_path)

# Set up seaborn styling
sns.set_style("whitegrid")

# Filter necessary columns for plotting
# Pivoting the data
heatmap_data_event_count = soc_results.pivot(index='SOC_ABBREV', columns='Drug', values='event_count')
heatmap_data_ror = soc_results.pivot(index='SOC_ABBREV', columns='Drug', values='ROR')
heatmap_data_prr = soc_results.pivot(index='SOC_ABBREV', columns='Drug', values='PRR')
heatmap_data_ci_lower = soc_results.pivot(index='SOC_ABBREV', columns='Drug', values='CI_lower')


# Plotting the heatmaps
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.1, wspace=0.1)

# Heatmap for event_count
sns.heatmap(heatmap_data_event_count, ax=axes[0, 0], cmap='YlGnBu', cbar_kws={'label': 'Event Count'}, robust=True)
axes[0, 0].set_title('Event Count Heatmap')
axes[0, 0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# Heatmap for ROR
sns.heatmap(heatmap_data_ror, ax=axes[0, 1], cmap='YlGnBu', cbar_kws={'label': 'ROR'}, robust=True)
axes[0, 1].set_title('ROR Heatmap')
axes[0, 1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# Heatmap for PRR
sns.heatmap(heatmap_data_prr, ax=axes[1, 0], cmap='YlGnBu', cbar_kws={'label': 'PRR'}, robust=True)
axes[1, 0].set_title('PRR Heatmap')
axes[1, 0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# Heatmap for CI_lower
sns.heatmap(heatmap_data_ci_lower, ax=axes[1, 1], cmap='YlGnBu', cbar_kws={'label': 'CI Lower Bound'}, robust=True)
axes[1, 1].set_title('CI Lower Bound Heatmap')
axes[1, 1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# Display the plots
plt.show()
