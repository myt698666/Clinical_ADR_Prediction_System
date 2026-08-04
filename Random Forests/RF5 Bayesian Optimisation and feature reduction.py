import pandas as pd
import optuna
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_score, accuracy_score, f1_score, matthews_corrcoef)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import time

# Load input and output files
input_file_path = "STITCH_Identifiers/drug_protein_interaction_matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

# Merge and clean data
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')
merged_data = merged_data.dropna()

# Select only the 5 targets of interest
selected_targets = ["Psych"]  # Change this if you want more targets

# Define feature matrix
X = merged_data.iloc[:, :-len(outputs.columns)]
output_columns = outputs.columns.drop('Drug')  # Exclude the 'Drug' column
results = {}

def objective(trial, target, X_train, y_train, X_test, y_test):
    # Hyperparameter optimization with Optuna
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

    smote = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    rf_model.fit(X_train_res, y_train_res)
    y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)

    return auc_score

# Iterative process for feature reduction and optimization
for target in selected_targets:
    print(f"Optimizing for target: {target}")

    start_time = time.time()
    results[target] = []
    # Initial feature set
    X_target = X.copy()
    y_target = merged_data[target]

    # Run Optuna optimization for 6 iterations, reducing features by 50% each time
    for iteration in range(6):
        print(f"Iteration {iteration + 1}")

        # Train-test split for this iteration
        X_train, X_test, y_train, y_test = train_test_split(X_target, y_target, test_size=0.2, random_state=42)

        # Print the number of features going into the model
        print(f"Number of features in this iteration: {X_train.shape[1]}")

        # Create and optimize the model using Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, target, X_train, y_train, X_test, y_test), n_trials=2)

        # Get the best model from the study and retrain it
        best_rf_model = RandomForestClassifier(**study.best_params, random_state=42)
        best_rf_model.fit(X_train, y_train)

        # Access feature importances from the best model
        importances = best_rf_model.feature_importances_

        # Sort features by importance and select top 50% features
        top_n = 0.5  # Keep top 50% of features
        indices = importances.argsort()[::-1]
        top_n_features = int(X_train.shape[1] * top_n)

        # Get the names of the top N features
        selected_features = X_train.columns[indices[:top_n_features]]

        # Create the feature importance DataFrame using the selected features' names
        feature_importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': importances[indices[:top_n_features]]})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Print the feature importance table for this iteration
        print(feature_importance_df)

        # Reduce the dataset to the selected features
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        X_target = X_target[selected_features]  # Update X_target to match the reduced features

        # Store the iteration result as a dictionary inside results[target]
        iteration_results = {
        'Iteration': iteration + 1,
        'Number of Features': X_train.shape[1]*2,
        'ROC AUC': study.best_value,
        'Best Parameters': study.best_params,
        }
        # Append the results of each iteration for this target
        results[target].append(iteration_results)


        print(f"Best ROC AUC for {target}: {study.best_value:.4f}")
        print(f"Best Parameters: {study.best_params}\n")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time for {target}: {total_time:.2f} seconds")

    # Save the final results
    with open(f"Random Forests/Best Models/FR_{target}_best_params.json", "w") as f:
        json.dump(results[target], f, indent=4)


# Optionally, save the final results for all targets
#with open("final_best_params.json", "w") as f:
    #json.dump(results, f, indent=4)

# Create a final model on the reduced feature set and evaluate performance
#final_results = []
#for target in selected_targets:
    #y_target = merged_data[target]
    #X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=42)

    # Use the best model from the last iteration of feature reduction
    #final_rf_model = RandomForestClassifier(**results[target]['Best Parameters'], random_state=42)
    #smote = SMOTETomek(random_state=42)
    #X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    #final_rf_model.fit(X_train_res, y_train_res)
    #y_pred_prob = final_rf_model.predict_proba(X_test)[:, 1]
    #y_pred = (y_pred_prob >= 0.5).astype(int)

    #auc = roc_auc_score(y_test, y_pred_prob)
    #precision = precision_score(y_test, y_pred, zero_division=0)
    #accuracy = accuracy_score(y_test, y_pred)
    #f1 = f1_score(y_test, y_pred, zero_division=0)
    #mcc = matthews_corrcoef(y_test, y_pred)

    #final_results.append({
        #'Target': target,
        #'ROC AUC': auc,
        #'Precision': precision,
        #'Accuracy': accuracy,
        #'F1 Score': f1,
        #'MCC': mcc,
    #})

    #print(f"Final model for {target} - AUC: {auc:.2f}, Precision: {precision:.2f}, Accuracy: {accuracy:.2f}, F1: {f1:.2f}, MCC: {mcc:.2f}")

# Plotting final results (optional)
#for result in final_results:
    #print(result)
