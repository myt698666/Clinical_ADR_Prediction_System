import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths
folder_path = "ADR_sex_age_analysis"
age_files = [f for f in os.listdir(folder_path) if f.startswith("matrix_AGE_10")]
sex_files = [f for f in os.listdir(folder_path) if f.startswith("matrix_SEX")]

# Load matrices
def load_matrices(file_list, folder_path):
    matrices = {}
    for file in file_list:
        matrix_name = file.split(".")[0]
        file_path = os.path.join(folder_path, file)
        matrices[matrix_name] = pd.read_csv(file_path, index_col=0, header=0)
    return matrices

age_matrices = load_matrices(age_files, folder_path)
sex_matrices = load_matrices(sex_files, folder_path)

# Visualise AGE matrices
plt.figure(figsize=(15, 10))
for i, (name, matrix) in enumerate(age_matrices.items()):
    plt.subplot(3, 5, i + 1)  # Adjust rows/cols based on number of matrices
    sns.heatmap(matrix, cbar=False, cmap="Blues", square=False)
    plt.title(name)
    plt.xticks(fontsize=0.2)  # Set x-axis tick font size to 0.2
    plt.yticks(fontsize=0.2)  # Set y-axis tick font size to 0.2
plt.show()

# Compare SEX matrices
sex_matrix_names = list(sex_matrices.keys())
if len(sex_matrix_names) == 2:
    sex_diff = abs(sex_matrices[sex_matrix_names[0]] - sex_matrices[sex_matrix_names[1]])
    plt.figure(figsize=(10, 5))
    sns.heatmap(sex_diff, cmap="Reds", square=False)
    plt.title(f"Difference: {sex_matrix_names[0]} vs {sex_matrix_names[1]}")
    plt.xticks(fontsize=0.2)  # Set x-axis tick font size to 8
    plt.yticks(fontsize=0.2)  # Set y-axis tick font size to 8
    plt.show()
else:
    print("Ensure exactly 2 SEX matrices for comparison.")
