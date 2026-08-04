import pandas as pd
import optuna
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_score, accuracy_score, f1_score, matthews_corrcoef)
from imblearn.over_sampling import SMOTE
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------
# Load and merge data
# ---------------------------
input_file_path = "STITCH_Identifiers/drug_protein_interaction_matrix.csv"
output_file_path = "ADR_Summary/HLGT_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')
merged_data = merged_data.dropna()

# ---------------------------
# Configuration
# ---------------------------
# Final evaluation for these selected targets
selected_targets = ["Psych", "Nerv", "Neopl", "Preg", "Endo"]

# Define feature matrix (all columns except the output columns)
X = merged_data.iloc[:, :-len(outputs.columns)]
output_columns = outputs.columns.drop('Drug')  # Exclude the 'Drug' column

# Folder for saving best parameters
best_params_folder = "Random Forests/Best Models"
os.makedirs(best_params_folder, exist_ok=True)

# List of targets for which best parameters have already been saved
processed_targets = [
    os.path.basename(f).replace("_best_params.json", "") 
    for f in glob.glob(os.path.join(best_params_folder, "*.json"))
]

# Determine remaining targets to optimize
remaining_targets = [t for t in output_columns if t not in processed_targets]

# Dictionary to record best parameters, results, and skipped targets
best_params = {}
results = {}
skipped_targets = {}

# ---------------------------
# Pre-filter: Remove targets with only one class overall
# ---------------------------
# We will remove any target (from the remaining targets) if its response variable
# has less than 2 unique values.
filtered_targets = []
for target in remaining_targets:
    unique_vals = np.unique(merged_data[target])
    if len(unique_vals) < 2:
        msg = f"Skipping target '{target}' because it only has one class: {unique_vals}."
        print(msg)
        results[target] = {"Error": msg}
        skipped_targets[target] = msg
    else:
        filtered_targets.append(target)

# Update remaining_targets with filtered list
remaining_targets = filtered_targets

# ---------------------------
# Define objective function for Optuna
# ---------------------------
def objective(trial, target):

    # Hyperparameter suggestions for the RandomForestClassifier
    n_estimators = trial.suggest_int('n_estimators', 10, 300)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 40)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 40)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=42
    )

    y = merged_data[target]
    # Use stratified splitting so that the training set has a similar class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Count the number of positive samples in the training set.
    positive_count = (y_train == 1).sum()
    
    if positive_count < 2:
            # Although we already removed targets with a single class overall,
            # a very imbalanced train/test split might still produce a training set
            # with only one class.
        print(f"Target '{target}': too few positive data points in training set ({positive_count}).")
        return 0.0
    elif positive_count < 5:
        smote = SMOTE(k_neighbors=min(5, np.min(y_train.value_counts()) - 1))
    else:
        smote = SMOTE(random_state=42)

    try:
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    except ValueError as e:
        print(f"Trial {trial}: Skipping due to SMOTE error - {e}")
        return 0.0  # Return a neutral score and exit the function
    
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    rf_model.fit(X_train_res, y_train_res)
    
    # Get predicted probabilities for the positive class
    y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
    
    # Check that y_test has both classes
    if len(np.unique(y_test)) < 2:
        print(f"Trial for target '{target}' has y_test with only one class: {np.unique(y_test)}. Returning default score.")
        return 0.0

    auc_score = roc_auc_score(y_test, y_pred_prob)
    
   
    return auc_score

# ---------------------------
# Bayesian Optimization for Remaining Targets
# ---------------------------
if not remaining_targets:
    print("No targets to optimize after filtering for multi-class responses.")
else:
    print(f"Resuming optimization for targets: {remaining_targets}")
    
    for target in remaining_targets:
        # Check overall positive count before any processing (if you wish to use that as an additional filter)
        overall_positive = (merged_data[target] == 1).sum()
        if overall_positive < 2:
            msg = f"Skipping optimization for target '{target}': too few positive data points overall ({overall_positive})."
            print(msg)
            results[target] = {'Error': msg}
            skipped_targets[target] = msg
            continue

        print(f"Optimising for target: {target}")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, target), n_trials=50)
        
        best_params[target] = study.best_params
        results[target] = {
            'Best ROC AUC': study.best_value,
            'Best Parameters': study.best_params
        }

        print(f"Best ROC AUC for {target}: {study.best_value:.4f}")
        print(f"Best Parameters: {study.best_params}\n")

        # Save the best parameters for this target
        with open(os.path.join(best_params_folder, f"{target}_best_params.json"), "w") as f:
            json.dump(best_params[target], f, indent=4)

    print("Optimization completed for remaining targets.")

# Optionally, save the combined best parameters (including those successfully optimized)
with open("best_params_HLGT.json", "w") as f:
    json.dump(best_params, f, indent=4)

# ---------------------------
# Final Evaluation for Selected Targets
# ---------------------------
final_results = []
feature_importances = {}

for target in selected_targets:
    best_params_path = os.path.join(best_params_folder, f"{target}_best_params.json")
    
    # Skip target if no optimization was done or if it was marked as having too few data points
    if not os.path.exists(best_params_path):
        msg = f"Skipping final evaluation for {target} as it was not optimized or had too few data points."
        print(msg)
        final_results.append({'Target': target, 'Error': msg})
        skipped_targets[target] = msg
        continue

    # Also check that the target's data still contains both classes
    if len(np.unique(merged_data[target])) < 2:
        msg = f"Skipping final evaluation for {target} because it only has one class."
        print(msg)
        final_results.append({'Target': target, 'Error': msg})
        skipped_targets[target] = msg
        continue

    with open(best_params_path, "r") as f:
        best_params[target] = json.load(f)

    y = merged_data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    positive_count = (y_train == 1).sum()
    if positive_count < 5:
        smote = SMOTE(random_state=42, k_neighbors=1)
    else:
        smote = SMOTE(random_state=42)
    
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    final_rf_model = RandomForestClassifier(**best_params[target], random_state=42)
    final_rf_model.fit(X_train_res, y_train_res)

    y_pred_prob = final_rf_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    final_results.append({
        'Target': target,
        'ROC AUC': auc,
        'Precision': precision,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'MCC': mcc,
        'y_test': y_test.values,
        'y_pred': y_pred
    })
    
    print(f"Final model for {target} - AUC: {auc:.2f}, Precision: {precision:.2f}, Accuracy: {accuracy:.2f}, F1: {f1:.2f}, MCC: {mcc:.2f}")

    importances = final_rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_importances[target] = feature_importance_df
    os.makedirs("Feature Importance", exist_ok=True)
    feature_importance_df.to_csv(f"Feature Importance/feature_importance_{target}.csv", index=False)

    print(f"Feature importance for {target} saved to 'Feature Importance/feature_importance_{target}.csv'.")

# ---------------------------
# Plotting Results
# ---------------------------
# Custom colormap for heatmaps
custom_cmap = LinearSegmentedColormap.from_list(
    "custom", ["#807163", "#F1F2F3", "#12616E"], N=256
)

plt.figure(figsize=(20, 10))
for idx, result in enumerate(final_results):
    if 'Error' in result:
        continue
    comparison_df = pd.DataFrame({'Actual': result['y_test'], 'Predicted': result['y_pred']})
    plt.subplot(2, 3, idx + 1)
    sns.heatmap(comparison_df.T, cmap=custom_cmap, cbar=True, xticklabels=False, 
                yticklabels=['Actual', 'Predicted'])
    plt.title(f"Heatmap for Target: {result['Target']}\nROC AUC: {result['ROC AUC']:.2f}")
plt.tight_layout()
plt.show()

for target in selected_targets:
    if target not in feature_importances:
        continue
    
    plt.figure(figsize=(10, 6))
    feature_importance_df = feature_importances[target]
    sns.barplot(x=feature_importance_df['Importance'][:10], color="#12616E", y=feature_importance_df['Feature'][:10])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top 10 Feature Importances for {target}")
    plt.show()

# ---------------------------
# Summary of Skipped Targets
# ---------------------------
if skipped_targets:
    print("\nThe following targets were skipped due to having only one class or insufficient data:")
    for target, message in skipped_targets.items():
        print(f"{target}: {message}")
else:
    print("\nNo targets were skipped due to insufficient or single-class data.")
