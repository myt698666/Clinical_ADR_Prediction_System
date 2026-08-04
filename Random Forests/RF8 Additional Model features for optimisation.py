import pandas as pd
import numpy as np
import optuna
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, f1_score, matthews_corrcoef
from sklearn.feature_selection import RFE, RFECV
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from matplotlib.colors import LinearSegmentedColormap

# Load input and output files
input_file_path = "STITCH_Identifiers/drug_protein_interaction_matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

# Merge and clean data
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')
merged_data = merged_data.dropna()

# Select only the targets of interest
selected_targets = ["Psych"]  # Add other targets as needed: , "Nerv", "Neopl", "Preg", "Endo"

# Define feature matrix
X = merged_data.iloc[:, :-len(outputs.columns)]
output_columns = outputs.columns.drop('Drug')  # Exclude the 'Drug' column

def reduce_features(X, y, n_features_to_select=1000):
    print("Starting feature reduction process...")
    
    # Remove highly correlated features
    print("Calculating correlation matrix...")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    print(f"Identified {len(to_drop)} highly correlated features to drop: {to_drop}")
    
    X_dropped = X.drop(to_drop, axis=1)
    print(f"Shape after dropping correlated features: {X_dropped.shape}")

    # Recursive Feature Elimination with Cross-Validation
    print("Starting Recursive Feature Elimination (RFECV)...")
    rfe = RFECV(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42), 
        step=50, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc', 
        n_jobs=-1
    )
    rfe.fit(X_dropped, y)
    print("RFECV completed.")
    num_selected = np.sum(rfe.support_)
    print(f"RFECV selected {num_selected} features out of {X_dropped.shape[1]}.")

    return X_dropped.iloc[:, rfe.support_], rfe.support_


# Objective function for Optuna
def objective(trial, X, y, model_type='rf'):
    if model_type == 'rf':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 300),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 40),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 40),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
        model = RandomForestClassifier(**params, random_state=42)
    else:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
        }
        model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline = Pipeline([
            ('smote', SMOTETomek(random_state=42)),
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred_prob = pipeline.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, y_pred_prob))

    return np.mean(scores)

# Main execution
results = {}
best_params = {}
final_results = []
feature_importances = {}

for target in selected_targets:
    print(f"Processing target: {target}")
    
    y = merged_data[target]
    X_reduced, feature_mask = reduce_features(X, y)
    
    for model_type in ['rf', 'xgb']:
        print(f"Optimizing {model_type.upper()} model")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_reduced, y, model_type), n_trials=100)
        
        best_params[f"{target}_{model_type}"] = study.best_params
        results[f"{target}_{model_type}"] = {
            'Best ROC AUC': study.best_value,
            'Best Parameters': study.best_params
        }

        print(f"Best {model_type.upper()} ROC AUC for {target}: {study.best_value:.4f}")
        print(f"Best Parameters: {study.best_params}\n")

        with open(f"Random Forests/Best Models/{target}_{model_type}_best_params.json", "w") as f:
            json.dump(study.best_params, f, indent=4)

    # Final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    
    for model_type in ['rf', 'xgb']:
        if model_type == 'rf':
            final_model = RandomForestClassifier(**best_params[f"{target}_rf"], random_state=42)
        else:
            final_model = XGBClassifier(**best_params[f"{target}_xgb"], random_state=42, use_label_encoder=False, eval_metric='logloss')

        pipeline = Pipeline([
            ('smote', SMOTETomek(random_state=42)),
            ('scaler', StandardScaler()),
            ('model', final_model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_pred_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        final_results.append({
            'Target': target,
            'Model': model_type.upper(),
            'ROC AUC': auc,
            'Precision': precision,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'MCC': mcc,
            'y_test': y_test.values,
            'y_pred': y_pred
        })
        
        print(f"Final {model_type.upper()} model for {target} - AUC: {auc:.2f}, Precision: {precision:.2f}, Accuracy: {accuracy:.2f}, F1: {f1:.2f}, MCC: {mcc:.2f}")

        if model_type == 'rf':
            importances = pipeline.named_steps['model'].feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': X_reduced.columns, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            feature_importances[f"{target}_{model_type}"] = feature_importance_df
            feature_importance_df.to_csv(f"Feature Importance/feature_importance_{target}_{model_type}.csv", index=False)
            print(f"Feature importance for {target} ({model_type.upper()}) saved to 'feature_importance_{target}_{model_type}.csv'.")

# Visualization
custom_cmap = LinearSegmentedColormap.from_list("custom", ["#807163", "#F1F2F3", "#12616E"], N=256)

plt.figure(figsize=(20, 10))
for idx, result in enumerate(final_results):
    comparison_df = pd.DataFrame({'Actual': result['y_test'], 'Predicted': result['y_pred']})
    plt.subplot(2, 3, idx + 1)
    sns.heatmap(comparison_df.T, cmap=custom_cmap, cbar=True, xticklabels=False, yticklabels=['Actual', 'Predicted'])
    plt.title(f"Heatmap for Target: {result['Target']} ({result['Model']})\nROC AUC: {result['ROC AUC']:.2f}")
plt.tight_layout()
plt.show()

for target in selected_targets:
    for model_type in ['rf', 'xgb']:
        if f"{target}_{model_type}" in feature_importances:
            plt.figure(figsize=(10, 6))
            feature_importance_df = feature_importances[f"{target}_{model_type}"]
            sns.barplot(x=feature_importance_df['Importance'][:10], color="#12616E", y=feature_importance_df['Feature'][:10])
            plt.xlabel("Feature Importance")
            plt.ylabel("Feature")
            plt.title(f"Top 10 Feature Importances for {target} ({model_type.upper()})")
            plt.show()
