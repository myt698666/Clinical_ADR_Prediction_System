import pandas as pd
import optuna
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import numpy as np

# Load input and output files
input_file_path = "STITCH_Identifiers/drug_protein_interaction_matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

# Merge and clean data
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')
merged_data = merged_data.dropna()

# Select only the 5 targets of interest
selected_targets = ["Nerv"]

# Define feature matrix
X = merged_data.iloc[:, :-len(outputs.columns)]

# List of random states to test
random_states = [10, 42, 99, 123, 2024]  # You can modify these

# Store results
all_results = {}

# Define an objective function for Optuna
def objective(trial, target, random_state):
    # Sample hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 300)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 40)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 40)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

    # Define model
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=42
    )

    # Get target data
    y = merged_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Apply SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train model
    rf_model.fit(X_train_res, y_train_res)

    # Evaluate model
    y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)

    return auc_score

# Iterate over random states
for random_state in random_states:
    print(f"Testing with random_state = {random_state}\n")
    best_params = {}
    results = {}

    for target in selected_targets:
        print(f"Optimising for target: {target}")
        
        # Create Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, target, random_state), n_trials=100)
        
        # Store best parameters
        best_params[target] = study.best_params
        results[target] = {
            'Best ROC AUC': study.best_value,
            'Best Parameters': study.best_params
        }

        print(f"Best ROC AUC for {target} with random_state {random_state}: {study.best_value:.4f}")
        print(f"Best Parameters: {study.best_params}\n")

    # Save best parameters for each random state
    with open(f"best_params_Nerv_random_state_{random_state}.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Best hyperparameters saved for random_state {random_state}.\n")

    # Train final models with best hyperparameters
    final_results = []
    feature_importances = {}

    for target in selected_targets:
        y = merged_data[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        # Apply SMOTE
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Train final model
        final_rf_model = RandomForestClassifier(**best_params[target], random_state=42)
        final_rf_model.fit(X_train_res, y_train_res)

        # Evaluate model
        y_pred_prob = final_rf_model.predict_proba(X_test)[:, 1]

        # Convert predictions to binary (0 or 1) using 0.5 threshold
        y_pred = (y_pred_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_pred_prob)

        final_results.append({
            'Target': target,
            'ROC AUC': auc,
            'y_test': y_test.values,
            'y_pred': y_pred
        })
        
        print(f"Final model trained for target '{target}' with random_state {random_state} and ROC AUC: {auc:.2f}")

        # Extract feature importance for positive classification
        importances = final_rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Store feature importance for later analysis
        feature_importances[target] = feature_importance_df

        # Save feature importances to a CSV file
        feature_importance_df.to_csv(f"feature_importance_{target}_random_state_{random_state}.csv", index=False)

        print(f"Feature importance for {target} with random_state {random_state} saved.\n")

    all_results[random_state] = final_results

# Generate heatmaps for each random state
plt.figure(figsize=(20, 10))
for rs_idx, random_state in enumerate(random_states):
    for idx, result in enumerate(all_results[random_state]):
        comparison_df = pd.DataFrame({
            'Actual': result['y_test'],
            'Predicted': result['y_pred']
        })

        plt.subplot(len(random_states), len(selected_targets), rs_idx * len(selected_targets) + idx + 1)
        sns.heatmap(
            comparison_df.T, 
            cmap='coolwarm', 
            cbar=True, 
            xticklabels=False, 
            yticklabels=['Actual', 'Predicted']
        )
        plt.title(f"Heatmap for Target: {result['Target']}\nRandom State: {random_state}\nROC AUC: {result['ROC AUC']:.2f}")

plt.tight_layout()
plt.show()

# Display feature importance plots
for random_state in random_states:
    for target in selected_targets:
        plt.figure(figsize=(10, 6))
        feature_importance_df = feature_importances[target]
        sns.barplot(x=feature_importance_df['Importance'][:10], y=feature_importance_df['Feature'][:10])
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title(f"Top 10 Feature Importances for {target}\nRandom State: {random_state}")
        plt.show()
