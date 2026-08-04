import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from imblearn.over_sampling import SMOTE

# Create the saved models folder if it doesn't exist
os.makedirs('saved_models/SOC', exist_ok=True)

# Load the input (top 500 protein matrix) and output (ADR significance matrix) files
input_file_path = "STITCH_Identifiers/drug_protein_interaction_matrix.csv"
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
results = []

# Iterate through each target output column
for target in output_columns:
    y = merged_data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to balance data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=X.columns)

    feature_counts = []
    roc_auc_scores = []  # Changed to store ROC AUC scores
    max_features_list = ['sqrt', 'log2']  # Different max_features settings

    for max_features in max_features_list:
        # Initialize RandomForestClassifier with varying max_features
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=10, min_samples_leaf=2, max_features=max_features, random_state=42)
        rf_model.fit(X_train_resampled_df, y_train_resampled)
    
        # Select top features based on importance
        num_features_to_keep = max(1, int(len(X_train_resampled_df.columns) * 0.9))
        feature_importances = pd.Series(rf_model.feature_importances_, index=X_train_resampled_df.columns)
        selected_features = feature_importances.nlargest(num_features_to_keep).index.tolist()
        feature_counts.append(len(selected_features))

        # Transform train and test data
        X_train_selected = X_train_resampled_df[selected_features]
        X_test_selected = X_test[selected_features]

        # Train and predict with selected features
        rf_model.fit(X_train_selected, y_train_resampled)
        y_pred = rf_model.predict(X_test_selected)

        # Evaluate model performance using ROC AUC
        roc_auc = roc_auc_score(y_test, y_pred)  # Using ROC AUC score
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        roc_auc_scores.append(roc_auc)

        # Store results
        results.append({
            'max_features': max_features,
            'Target': target,
            'Features': len(selected_features),
            'ROC AUC': roc_auc,  # Store ROC AUC score
            'Precision': precision,
            'Recall': recall
        })

        print(f"max_features: {max_features}, ROC AUC: {roc_auc}")

    # Check the contents of roc_auc_scores
    print("ROC AUC Scores:", roc_auc_scores)

    # Filter out None values from max_features_list and roc_auc_scores
    filtered_max_features = [x for x in max_features_list if x is not None]
    filtered_roc_auc_scores = [roc_auc for roc_auc, max_f in zip(roc_auc_scores, max_features_list) if max_f is not None]

    # Plot max_features vs ROC AUC using a bar chart
    if filtered_roc_auc_scores:  # Ensure filtered_roc_auc_scores is not empty
        plt.figure(figsize=(10, 5))
        plt.bar(filtered_max_features, filtered_roc_auc_scores, color='blue')
        plt.xlabel("Max Features")
        plt.ylabel("ROC AUC Score")
        plt.title(f"Impact of max_features on ROC AUC Score for Target: {target}")
        plt.show()
    else:
        print("Error: ROC AUC scores are empty after filtering None values.")

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv("Random Forests/Results/max_features_results.csv", index=False)
