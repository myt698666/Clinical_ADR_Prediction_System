import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
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
    
    class_weights_list = [None, 'balanced', 'balanced_subsample', {0: 1, 1: 2}, {0: 1, 1: 3}]
    use_smote_options = [False, True]
    
    for use_smote in use_smote_options:
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=X.columns)
        else:
            X_train_resampled_df, y_train_resampled = X_train, y_train
    
        for class_weight in class_weights_list:
            # Initialize RandomForestClassifier with varying class weights
            rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=10, 
                                              min_samples_leaf=2, max_features='sqrt', class_weight=class_weight, 
                                              random_state=42)
            rf_model.fit(X_train_resampled_df, y_train_resampled)
            
            # Select top features based on importance
            num_features_to_keep = max(1, int(len(X_train_resampled_df.columns) * 0.9))
            feature_importances = pd.Series(rf_model.feature_importances_, index=X_train_resampled_df.columns)
            selected_features = feature_importances.nlargest(num_features_to_keep).index.tolist()

            # Transform train and test data
            X_train_selected = X_train_resampled_df[selected_features]
            X_test_selected = X_test[selected_features]
            
            # Train and predict with selected features
            rf_model.fit(X_train_selected, y_train_resampled)
            y_pred_proba = rf_model.predict_proba(X_test_selected)[:, 1]  # Get probability for the positive class

            # Evaluate model performance using ROC metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)

            # Store results
            results.append({
                'Target': target,
                'Use_SMOTE': use_smote,
                'Class_Weight': class_weight,
                'Features': len(selected_features),
                'ROC AUC': roc_auc,
                'PR AUC': pr_auc
            })
            
            print(f"Use_SMOTE: {use_smote}, Class_Weight: {class_weight}, ROC AUC: {roc_auc}, PR AUC: {pr_auc}")

            # Plot ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for Target: {target}')
            plt.legend()
            plt.show()
    
    # Save the final model with the best class weight setting
    model_filename = f"saved_models/SOC/random_forest_best_{target}_roc.pkl"
    joblib.dump(rf_model, model_filename)

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv("Random Forests/Results/class_weight_comparison_roc_results.csv", index=False)
