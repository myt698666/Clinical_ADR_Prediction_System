import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

# Create the saved models folder if it doesn't exist
os.makedirs('saved_models/HLGT', exist_ok=True)

# Load the input (top 500 protein matrix) and output (ADR significance matrix) files
input_file_path = "STITCH_Identifiers/top_500_protein_drug_interaction_matrix.csv"
output_file_path = "ADR_Summary/HLGT_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

# Perform a left join on the 'Drug' column
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')

# Drop rows with missing output values (for all target columns)
merged_data = merged_data.dropna()

# Separate features (inputs) and target columns
X = merged_data.iloc[:, :-len(outputs.columns)]  # Inputs (all columns in `inputs`)
output_columns = outputs.columns.drop('Drug')    # Exclude the 'Drug' column
results = []

# Train 5 models for each target column and use majority vote for the final prediction
for target in output_columns:
    y = merged_data[target]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize an array to store predictions from the 5 models
    all_predictions = []
    
    # List to hold trained models
    trained_models = []
    
    for random_state in [10, 20, 42, 57, 83]:  # Train 5 models with different random states (1 to 5)
        rf_model = RandomForestClassifier(random_state=random_state)
        rf_model.fit(X_train, y_train)
        
        # Sanitize the model filename by replacing '/' with '_'
        model_filename = f"saved_models/HLGT/random_forest_model_{target}_{random_state}.pkl"
        sanitized_model_filename = model_filename.replace('/', '_')  # Replace '/' with '_'

        # Save the trained model to a file for later use in 'saved_models/HLGT' folder
        joblib.dump(rf_model, sanitized_model_filename)
        
        trained_models.append(rf_model)
        
        # Predict using the trained model
        y_pred = rf_model.predict(X_test)
        all_predictions.append(y_pred)
    
    # Apply majority vote: If 2 or more models predict 1, the final prediction is 1
    majority_vote = np.sum(all_predictions, axis=0) >= 2  # Majority rule (at least 2 models should predict 1)
    majority_vote = majority_vote.astype(int)  # Convert boolean to 1 or 0
    
    # Evaluate accuracy with majority voting
    accuracy = accuracy_score(y_test, majority_vote)
    results.append({
        'Target': target,
        'Accuracy': accuracy,
        'y_test': y_test.values,
        'y_pred': majority_vote,
        'models': trained_models
    })
    print(f"Model trained for target '{target}' with accuracy: {accuracy:.2f}")

# Sort results by accuracy and select the top 10 accurate models
top_10_results = sorted(results, key=lambda x: x['Accuracy'], reverse=True)[:10]

# Predict with the top 10 models and visualize the validation results (y_pred vs y_test)
plt.figure(figsize=(20, 15))
for idx, result in enumerate(top_10_results):
    # Compare the predicted values to the actual ones (y_test) for this target
    comparison_df = pd.DataFrame({
        'Actual': result['y_test'],
        'Predicted': result['y_pred']
    })
    
    plt.subplot(2, 5, idx + 1)  # 2 rows, 5 columns for 10 plots
    sns.heatmap(
        comparison_df.T, 
        cmap='coolwarm', 
        cbar=True, 
        xticklabels=False, 
        yticklabels=['Actual', 'Predicted']
    )
    plt.title(f"Heatmap for Target: {result['Target']}\nAccuracy: {result['Accuracy']:.2f}")

plt.tight_layout()
plt.show()