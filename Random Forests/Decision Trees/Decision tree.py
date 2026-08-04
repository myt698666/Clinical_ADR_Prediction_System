import pandas as pd
import optuna
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_score, accuracy_score, f1_score, matthews_corrcoef,recall_score)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek   # Changed import
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import time
import os

# Load input and output files
input_file_path = "STITCH_Identifiers/drug_protein_interaction_matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

# Merge and clean data
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')
merged_data = merged_data.dropna()

# Select only the 5 targets of interest
selected_targets = ["Psych"]#, "Nerv", "Neopl", "Preg", "Endo"]

# Define feature matrix
X = merged_data.iloc[:, :-len(outputs.columns)]
output_columns = outputs.columns.drop('Drug')    # Exclude the 'Drug' column
results = {}

def objective(trial, target):
    max_depth = trial.suggest_int('max_depth', 2, 40)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 40)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 40)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    dt_model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )

    y = merged_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    smote = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    dt_model.fit(X_train_res, y_train_res)
    y_pred_prob = dt_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    auc_score = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    return auc_score

best_params = {}

for target in output_columns:#selected_targets:
    print(f"Optimising for target: {target}")

    start_time = time.time()

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, target), n_trials=50)
    
    end_time = time.time()
    total_time = end_time-start_time
    print(total_time)
    
    best_params[target] = study.best_params
    results[target] = {
        'Best ROC AUC': study.best_value,
        'Best Parameters': study.best_params
    }

    print(f"Best ROC AUC for {target}: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}\n")

    output_dir = "Random Forests/Decision Trees/Best Models"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    file_path = os.path.join(output_dir, f"{target}_best_params.json")

    with open(file_path, "w") as f:
        json.dump(best_params, f, indent=4)

with open("decision_trees_best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

final_results = []
feature_importances = {}

for target in output_columns:#selected_targets:
    y = merged_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    final_dt_model = DecisionTreeClassifier(**best_params[target], random_state=42)
    final_dt_model.fit(X_train_res, y_train_res)

    y_pred_prob = final_dt_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    recall = recall_score(y_test,y_pred)

    final_results.append({
        'Target': target,
        'ROC AUC': auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall':recall,
        'F1 Score': f1,
        'MCC': mcc,
        'y_test': y_test.values,
        'y_pred': y_pred
    })
    # Save final results to CSV
    results_df = pd.DataFrame(final_results)
    results_output_path = "Random Forests/Results/Decision_Tree_metrics.csv"
    os.makedirs(os.path.dirname(results_output_path), exist_ok=True)  # Ensure the directory exists
    results_df.drop(columns=['y_test', 'y_pred']).to_csv(results_output_path, index=False)
    print(f"Final evaluation metrics saved to '{results_output_path}'")

    print(f"Final model for {target} - AUC: {auc:.2f}, Precision: {precision:.2f}, Accuracy: {accuracy:.2f}, F1: {f1:.2f}, MCC: {mcc:.2f}")

    importances = final_dt_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_importances[target] = feature_importance_df
    feature_importance_df.to_csv(f"Random Forests/Decision Trees/Feature Importance/feature_importance_{target}.csv", index=False)

    print(f"Feature importance for {target} saved to 'feature_importance_{target}.csv'.")
# Define custom colormap
custom_cmap = LinearSegmentedColormap.from_list(
    "custom", ["#807163", "#F1F2F3", "#12616E"], N=256
)


# Sort results by ROC AUC (or any other key metric)
final_results_sorted = sorted(final_results, key=lambda x: x['ROC AUC'], reverse=True)

# Select top 6 results
top_results = final_results_sorted[:6]

plt.figure(figsize=(20, 10))
for idx, result in enumerate(top_results):
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
