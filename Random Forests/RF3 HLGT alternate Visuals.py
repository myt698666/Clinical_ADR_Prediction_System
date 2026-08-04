import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Load the input and output files
input_file_path = "STITCH_Identifiers/top_500_protein_drug_interaction_matrix.csv"
output_file_path = "ADR_Summary/HLGT_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

# Merge and preprocess data
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')
merged_data = merged_data.dropna()
X = merged_data.iloc[:, :-len(outputs.columns)]  # Features
output_columns = outputs.columns.drop('Drug')    # Exclude 'Drug' column

# Directory to save models
model_save_dir = "saved_models/"
os.makedirs(model_save_dir, exist_ok=True)

results = []
for target in output_columns[:30]:
    y = merged_data[target]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize an array to store predictions from the 5 models
    all_predictions = []
    
    for random_state in [10, 20, 42, 57, 83]:  # Train 5 models with different random states
        rf_model = RandomForestClassifier(random_state=random_state)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        all_predictions.append(y_pred)
    
    # Apply majority vote: If 2 or more models predict 1, the final prediction is 1
    majority_vote = np.sum(all_predictions, axis=0) >= 2  # Majority rule (at least 2 models should predict 1)
    majority_vote = majority_vote.astype(int)  # Convert boolean to 1 or 0
    
    # Store the results for all models (regardless of predictions)
    results.append({
        'Target': target,
        'Accuracy': accuracy_score(y_test, majority_vote),
        'y_test': y_test.values,
        'y_pred': majority_vote
    })

# Filter results to include only those with at least one '1' in the predictions
filtered_results = [result for result in results if np.sum(result['y_pred']) > 0]

# Sort filtered results by accuracy and select the top models
top_5_results = sorted(filtered_results, key=lambda x: x['Accuracy'], reverse=True)[:5]

# Visualize confusion matrices for the top models
plt.figure(figsize=(20, 10))
for idx, result in enumerate(top_5_results):
    cm = confusion_matrix(result['y_test'], result['y_pred'], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    
    plt.subplot(2, 3, idx + 1)
    disp.plot(ax=plt.gca(), cmap='coolwarm', colorbar=False)
    plt.title(f"Confusion Matrix for Target: {result['Target']}\nAccuracy: {result['Accuracy']:.2f}")

plt.tight_layout()
plt.show()

