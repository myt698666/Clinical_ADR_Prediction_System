import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score

# Load the dataset
file_path = 'Drug Analysis.xlsx'
sheet_name = 'Final With Norm'
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Specify input columns (excluding 'Protein')
input_columns = [
    "Molecular Weight", "AlogP", "Aromatic Rings", "Cx LogD", "Cx LogP",
    "Cx Most Bpka", "HBA", "HBD", "HBA Lipinski", "HBD Lipinski",
    "Heavy Atoms", "Rotatable Bonds", "TPSA", "Confidence"
]
target_column = "ADR Norm (Normalized)"

# Filter the dataset
X = data[input_columns]
y = data[target_column]
proteins = data['Protein']
preferred_names = data['Preferred Name']
adr_categories = data['ADR'].unique()  # Use 'ADR' column for categories

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
proteins_test = proteins.loc[y_test.index]
preferred_names_test = preferred_names.loc[y_test.index]

# Create a dictionary to store models and results
models = {}
results_all = {}

# Train a separate model for each ADR category
for adr in adr_categories:
    print(f"Processing ADR: {adr}")
    
    # Filter data for this ADR
    mask = data['ADR'] == adr
    X_adr = X[mask]
    y_adr = y[mask]
    proteins_adr = proteins[mask]
    preferred_names_adr = preferred_names[mask]
    
    # Split the data for this ADR
    X_train_adr, X_test_adr, y_train_adr, y_test_adr = train_test_split(
        X_adr, y_adr, test_size=0.2, random_state=42
    )
    proteins_test_adr = proteins_adr.loc[y_test_adr.index]
    preferred_names_test_adr = preferred_names_adr.loc[y_test_adr.index]
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_adr, y_train_adr)
    models[adr] = model
    
    # Make predictions
    y_pred_adr = model.predict(X_test_adr)
    
    # Store results
    results = pd.DataFrame({
        'Protein': proteins_test_adr.values,
        'Preferred Name': preferred_names_test_adr.values,
        'Prediction': y_pred_adr
    })
    
    if not results.empty:  # Check if results dataframe is not empty
        results = results[results['Prediction'] > 0.05]  # Filter based on prediction threshold
        results_all[adr] = results
        
        # Pivot the data for heatmaps
        if not results.empty:
            heatmap_data = results.pivot_table(
                index='Protein',
                columns='Preferred Name',
                values='Prediction',
                aggfunc=np.mean,
                fill_value=0
            )

            # Plot the heatmaps
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                heatmap_data,
                cmap='viridis',
                cbar_kws={'label': 'Prediction Value'},
                annot=False,
                ax=ax
            )
            ax.set_title(f"ADR: {adr}")
            ax.set_xlabel('Preferred Name')
            ax.set_ylabel('Protein')
            plt.tight_layout()
            plt.show()
