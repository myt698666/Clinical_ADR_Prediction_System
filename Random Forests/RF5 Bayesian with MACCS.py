import pandas as pd
import optuna
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_score, accuracy_score, f1_score, matthews_corrcoef)
from imblearn.combine import SMOTETomek
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import time

# Load input and output files
input_file_1 = "STITCH_Identifiers/drug_protein_interaction_matrix.csv"
input_file_2 = "MACCS/MACCS_Keys_Output.csv"  # The new cleaned input file
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs_1 = pd.read_csv(input_file_1, index_col=0)
inputs_2 = pd.read_csv(input_file_2, index_col=0)
outputs = pd.read_csv(output_file_path)

# Merge input files on their common index (Drug)
inputs = inputs_1.merge(inputs_2, how='inner', left_index=True, right_index=True)

# Merge with outputs
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')
merged_data = merged_data.dropna()

# Select only the 5 targets of interest
selected_targets = ["Psych"]

# Define feature matrix
X = merged_data.iloc[:, :-len(outputs.columns)]
output_columns = outputs.columns.drop('Drug')
results = {}


def objective(trial, target):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    smote = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    rf_model.fit(X_train_res, y_train_res)
    y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    return roc_auc_score(y_test, y_pred_prob)


best_params = {}
for target in selected_targets:
    print(f"Optimising for target: {target}")
    start_time = time.time()

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, target), n_trials=50)
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
    best_params[target] = study.best_params
    results[target] = {
        'Best ROC AUC': study.best_value,
        'Best Parameters': study.best_params
    }
    
    with open(f"Random Forests/Best Models/MACCS_{target}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

final_results = []
feature_importances = {}
for target in selected_targets:
    y = merged_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    final_rf_model = RandomForestClassifier(**best_params[target], random_state=42)
    final_rf_model.fit(X_train_res, y_train_res)

    y_pred_prob = final_rf_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    final_results.append({
        'Target': target,
        'ROC AUC': roc_auc_score(y_test, y_pred_prob),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'y_test': y_test.values,
        'y_pred': y_pred
    })
    
    importances = final_rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_importances[target] = feature_importance_df
    feature_importance_df.to_csv(f"Feature Importance/feature_importance_MACCS_{target}.csv", index=False)

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
    sns.barplot(x=feature_importance_df['Importance'][:10], y=feature_importance_df['Feature'][:10], color="#12616E")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top 10 Feature Importances for {target}")
    plt.show()

