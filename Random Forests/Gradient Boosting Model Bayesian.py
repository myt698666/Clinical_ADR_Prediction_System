import pandas as pd
import optuna
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_score, accuracy_score, f1_score, matthews_corrcoef)
from imblearn.over_sampling import SMOTE
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Load input and output files
input_file_path = "STITCH_Identifiers/drug_protein_interaction_matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

# Merge and clean data
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')
merged_data = merged_data.dropna()

# Select only the 5 targets of interest
selected_targets = ["Psych"]  # , "Nerv", "Neopl", "Preg", "Endo"]

# Define feature matrix
X = merged_data.iloc[:, :-len(outputs.columns)]
output_columns = outputs.columns.drop('Drug')    # Exclude the 'Drug' column
results = {}

def objective(trial, target):
    # Define the 5 hyperparameters to optimize for Gradient Boosting:
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 40)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 40)
    
    gbm_model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    y = merged_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    gbm_model.fit(X_train_res, y_train_res)
    y_pred_prob = gbm_model.predict_proba(X_test)[:, 1]
    # Using 0.5 as threshold for classification
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    auc_score = roc_auc_score(y_test, y_pred_prob)
    return auc_score

best_params = {}

for target in selected_targets:  # or iterate over output_columns if desired
    print(f"Optimising for target: {target}")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, target), n_trials=100)
    
    best_params[target] = study.best_params
    results[target] = {
        'Best ROC AUC': study.best_value,
        'Best Parameters': study.best_params
    }
    
    print(f"Best ROC AUC for {target}: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}\n")
    
    with open(f"Random Forests/Best Models/GB/{target}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

final_results = []
feature_importances = {}

for target in selected_targets:
    y = merged_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=0)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    final_gbm_model = GradientBoostingClassifier(**best_params[target], random_state=42)
    final_gbm_model.fit(X_train_res, y_train_res)
    
    y_pred_prob = final_gbm_model.predict_proba(X_test)[:, 1]
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
    
    importances = final_gbm_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_importances[target] = feature_importance_df
    feature_importance_df.to_csv(f"Feature Importance/GB/feature_importance_{target}.csv", index=False)
    
    print(f"Feature importance for {target} saved to 'feature_importance_{target}.csv'.")

# Define custom colormap
custom_cmap = LinearSegmentedColormap.from_list(
    "custom", ["#807163", "#F1F2F3", "#12616E"], N=256
)

plt.figure(figsize=(20, 10))
for idx, result in enumerate(final_results):
    comparison_df = pd.DataFrame({'Actual': result['y_test'], 'Predicted': result['y_pred']})
    plt.subplot(2, 3, idx + 1)
    sns.heatmap(comparison_df.T, cmap=custom_cmap, cbar=True, xticklabels=False, yticklabels=['Actual', 'Predicted'])
    plt.title(f"Heatmap for Target: {result['Target']}\nROC AUC: {result['ROC AUC']:.2f}")
plt.tight_layout()
plt.show()

for target in selected_targets:
    plt.figure(figsize=(10, 6))
    feature_importance_df = feature_importances[target]
    sns.barplot(x=feature_importance_df['Importance'][:10], color="#12616E", y=feature_importance_df['Feature'][:10])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top 10 Feature Importances for {target}")
    plt.show()
