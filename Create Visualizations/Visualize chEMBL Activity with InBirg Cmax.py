import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the matrix and set the appropriate index
file_path = "Drug Activity/Drug_protein_interaction_pivot_matrix.csv"
data = pd.read_csv(file_path)

# Set 'Drug-Name' as the index
data.set_index('Drug-Name', inplace=True)

# Plotting the heatmap with inverted Spectral colors
plt.figure(figsize=(15, 10))
sns.heatmap(data, cmap='inferno', cbar=True, xticklabels=False, yticklabels=False)

# Titles and labels
plt.title('Drug-Protein Interaction Heatmap (Inverted Spectral Colors)')
plt.xlabel('Proteins')
plt.ylabel('Matched Drugs')

# Show plot
plt.show()
