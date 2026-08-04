import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
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

    # Initialize RandomForestClassifier without GridSearch
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, 
                                      min_samples_leaf=2, max_features='sqrt', random_state=42)
    rf_model.fit(X_train_resampled_df, y_train_resampled)
    
    feature_counts = []
    roc_auc_scores = []
    selected_features = X_train_resampled_df.columns
    
    for iteration in range(50):
        # Compute the number of features to retain (90% of the current feature count)
        num_features_to_keep = max(1, int(len(selected_features) * 0.9))
    
        # Sort features by importance and keep only the top ranked ones
        feature_importances = pd.Series(rf_model.feature_importances_, index=selected_features)
        selected_features = feature_importances.nlargest(num_features_to_keep).index.tolist()
        feature_counts.append(len(selected_features))
    
        if len(selected_features) == 0:
            break  # Stop if no features are left

        # Transform train and test data
        X_train_selected = X_train_resampled_df[selected_features]
        X_test_selected = X_test[selected_features]

        # Train and predict with reduced features
        rf_model.fit(X_train_selected, y_train_resampled)
        y_pred_proba = rf_model.predict_proba(X_test_selected)[:, 1]

        # Compute ROC AUC score
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        roc_auc_scores.append(roc_auc)

        # Store results
        results.append({
            'Iteration': iteration + 1,
            'Target': target,
            'Features': len(selected_features),
            'ROC AUC': roc_auc
        })
        
        print(f"Iteration {iteration + 1} - Target: {target} - ROC AUC: {roc_auc:.4f}")

    # Save the final reduced model
    model_filename = f"saved_models/SOC/random_forest_best_{target}_reduced.pkl"
    joblib.dump(rf_model, model_filename)

    # Plot feature count vs ROC AUC score
    plt.figure(figsize=(10, 5))
    plt.plot(feature_counts, roc_auc_scores, marker='o', linestyle='-')
    plt.xlabel("Number of Features")
    plt.ylabel("ROC AUC Score")
    plt.title(f"Feature Reduction Impact on ROC AUC Score for Target: {target}")
    plt.show()

    # Plot ROC Curve for the best model
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Target: {target}')
    plt.legend(loc='lower right')
    plt.show()

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv("Random Forests/Results/feature_reduction_results.csv", index=False)
