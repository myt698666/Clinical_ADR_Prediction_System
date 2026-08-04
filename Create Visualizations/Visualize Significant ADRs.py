import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Load the results data from the CSV file
file_path = 'ADR_Summary/soc_disproportionality_results.csv'
soc_results = pd.read_csv(file_path)

# Set up seaborn styling
sns.set_style("whitegrid")

# Pivoting the data
heatmap_data_event_count = soc_results.pivot(index='SOC_ABBREV', columns='Drug', values='event_count')
heatmap_data_ror = soc_results.pivot(index='SOC_ABBREV', columns='Drug', values='ROR')
heatmap_data_prr = soc_results.pivot(index='SOC_ABBREV', columns='Drug', values='PRR')
heatmap_data_ci_lower = soc_results.pivot(index='SOC_ABBREV', columns='Drug', values='CI_lower')

# Define custom colormap
custom_cmap = ListedColormap(['#D3D3D3', '#FF0000'])  # Grey for below, Red for above

# Create masks for thresholds
ror_mask = (heatmap_data_ror >= 2).astype(int)
prr_mask = (heatmap_data_prr >= 2).astype(int)
ci_lower_mask = (heatmap_data_ci_lower >= 1).astype(int)

# Plotting the heatmaps
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Heatmap for event_count
sns.heatmap(
    heatmap_data_event_count, ax=axes[0, 0], cmap='YlGnBu',
    cbar_kws={'label': 'Event Count'}, robust=True
)
axes[0, 0].set_title('Event Count Heatmap')
axes[0, 0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# Heatmap for ROR
sns.heatmap(
    ror_mask, ax=axes[0, 1], cmap=custom_cmap,
    cbar=False
)
axes[0, 1].set_title('ROR Threshold Heatmap')
axes[0, 1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# Heatmap for PRR
sns.heatmap(
    prr_mask, ax=axes[1, 0], cmap=custom_cmap,
    cbar=False
)
axes[1, 0].set_title('PRR Threshold Heatmap')
axes[1, 0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# Heatmap for CI Lower
sns.heatmap(
    ci_lower_mask, ax=axes[1, 1], cmap=custom_cmap,
    cbar=False
)
axes[1, 1].set_title('CI Lower Threshold Heatmap')
axes[1, 1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# Display the plots
plt.show()
