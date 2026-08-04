import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

# Load the input (top 500 protein matrix) and output (ADR significance matrix) files
input_file_path = "STITCH_Identifiers/top_500_protein_drug_interaction_matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

# Perform a left join on the 'Drug' column
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')

# Drop rows with missing output values (for all target columns)
merged_data = merged_data.dropna()

# Separate features (inputs) and target columns
X = merged_data.iloc[:, :-len(outputs.columns)]  # Inputs (all columns in `inputs`)
output_columns = outputs.columns.drop('Drug')    # Exclude the 'Drug' column

# Initialize a list to hold target accuracies and their corresponding predictions
target_predictions = []

# Iterate over the targets to make predictions with all 5 models and process results
for target in output_columns:
    all_predictions = []
    
    for random_state in [10, 20, 42, 57, 83]:
        model_filename = f"saved_models/SOC/random_forest_model_{target}_{random_state}.pkl"
        if os.path.exists(model_filename):  # Check if model file exists
            model = joblib.load(model_filename)
            
            # Make predictions using the loaded model on the full input data
            predictions = model.predict(X)
            all_predictions.append(predictions)
    
    # If predictions were collected for this target, process them
    if all_predictions:
        # Convert predictions to a numpy array (shape: [n_samples, 5])
        all_predictions = np.array(all_predictions).T  # Transpose to get [n_samples, 5]
        
        # Apply the rule: if 2 or more models predict 1, the final prediction is 1; else, 0
        final_predictions = np.sum(all_predictions, axis=1) >= 2  # This checks if 2 or more 1's appear
        
        # Convert the boolean result into integers (0 or 1)
        final_predictions = final_predictions.astype(int)
        
        # Store the final predictions for this target
        target_predictions.append({'Target': target, 'Final_Predictions': final_predictions})

# Sort the targets based on the average of the predicted values (if needed)
# For example, sorting by the mean of the final predictions
target_predictions_sorted = sorted(target_predictions, key=lambda x: np.mean(x['Final_Predictions']), reverse=True)

# Select the top 10 targets based on the averaged predictions
top_10_targets = [t['Target'] for t in target_predictions_sorted[:10]]

# Create a new figure for visualizations
plt.figure(figsize=(20, 15))

# Predict with the processed predictions and visualize the results
for idx, target in enumerate(top_10_targets):  # Only top 10 targets
    # Get the final predictions for this target
    final_predictions = next(t['Final_Predictions'] for t in target_predictions if t['Target'] == target)
    
    # Get the actual values (from the merged data)
    actual_values = merged_data[target].values
    
    # Ensure both actual_values and final_predictions are numeric
    actual_values = pd.to_numeric(actual_values, errors='coerce')
    final_predictions = pd.to_numeric(final_predictions, errors='coerce')
    
    # Create a DataFrame to compare actual vs predicted values
    comparison_df = pd.DataFrame({
        'Actual': actual_values,
        'Predicted': final_predictions
    })
    
    # Create a heatmap for this comparison
    plt.subplot(2, 5, idx + 1)  # 2 rows, 5 columns for 10 plots
    sns.heatmap(
        comparison_df.T, 
        cmap='coolwarm', 
        cbar=True, 
        xticklabels=False, 
        yticklabels=['Actual', 'Predicted']
    )
    plt.title(f"{target}")

plt.tight_layout()
plt.show()
