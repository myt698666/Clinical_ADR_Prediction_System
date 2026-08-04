import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, roc_curve
import joblib
import ast
import matplotlib.pyplot as plt

# Define paths
results_dir = "Random Forests/Results"
optimised_models_dir = "saved_models/Optimised_Models"
input_file_path = "STITCH_Identifiers/drug_protein_interaction_matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

# Ensure output directory exists
os.makedirs(optimised_models_dir, exist_ok=True)

# Load input and output data
inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

# Perform a left join on the 'Drug' column
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')

# Drop rows with missing output values (for all target columns)
merged_data = merged_data.dropna()

# Separate features (inputs) and target columns
X = merged_data.iloc[:, :-len(outputs.columns)]  # Inputs (all columns in `inputs`)
output_columns = outputs.columns.drop('Drug')    # Exclude the 'Drug' column

# Function to get top 5% best parameters for each target
def get_best_parameters(results_dir):
    best_params = {}
    for file in os.listdir(results_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(results_dir, file))
            metric = None
            if "ROC AUC" in df.columns:
                metric = "ROC AUC"
            elif "F1-Score" in df.columns:
                metric = "F1-Score"
            
            if metric and "Target" in df.columns:
                df = df.sort_values(by=metric, ascending=False)
                top_5_percent = int(len(df) * 0.05) or 1  # Ensure at least one entry
                best_rows = df.head(top_5_percent)
                
                for _, row in best_rows.iterrows():
                    target = row["Target"]
                    if target not in best_params:
                        best_params[target] = []
                    best_params[target].append(row.to_dict())
    return best_params

# Load top 5% best parameters
best_params = get_best_parameters(results_dir)

# Train and evaluate optimised models with GridSearchCV
for target in output_columns:
    if target not in best_params:
        continue  # Skip targets without optimised parameters
    
    print(best_params[target])  # Inspect the data structure
    
    y = merged_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        "n_estimators": list(set(int(p["n_estimators"]) for p in best_params[target])),
        "max_features": list(set(p["max_features"] for p in best_params[target])),
        "min_samples_leaf": list(set(int(p["min_samples_leaf"]) for p in best_params[target])),
        "min_samples_split": list(set(int(p["min_samples_split"]) for p in best_params[target])),
        "class_weight": list(set(
            ast.literal_eval(p["Class_Weight"]) if isinstance(p["Class_Weight"], str) and "{" in p["Class_Weight"] else p["Class_Weight"]
            for p in best_params[target]
        )),
    }
    
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
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

    # Save model
    model_path = os.path.join(optimised_models_dir, f"RF_{target}.pkl")
    joblib.dump(best_model, model_path)
    
    print(f"Optimised model for {target} saved. ROC AUC: {roc_auc:.4f}")
