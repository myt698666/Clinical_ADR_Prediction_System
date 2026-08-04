import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the matrix and set the appropriate index
file_path = "STITCH_Identifiers/top_500_protein_drug_interaction_matrix.csv"
data = pd.read_csv(file_path)

# Set 'Matched Drug' as the index
data.set_index('Matched Drug', inplace=True)

# Mask for zero values
mask = data == 0

# Plotting the heatmap with inverted Spectral colors
plt.figure(figsize=(15, 10))
sns.heatmap(data, mask=mask, cmap='binary', cbar=False, xticklabels=False, yticklabels=False)

# Titles and labels
plt.title('Drug-Protein Interaction Heatmap (Inverted Spectral Colors)')
plt.xlabel('Proteins')
plt.ylabel('Matched Drugs')

# Show plot
plt.show()
