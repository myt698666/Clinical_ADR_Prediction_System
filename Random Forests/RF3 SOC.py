import pandas as pd
import numpy as np
import os
import joblib
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# Create directory for model persistence
os.makedirs('saved_models/SOC', exist_ok=True)

# Load multi-modal feature matrix and significance targets
input_file_path = "../STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv"
output_file_path = "../ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col='Matched Drug')
outputs = pd.read_csv(output_file_path)

# Merge feature matrix with target significance matrix
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug').dropna()

# Segregate features and target outcomes
X = merged_data.iloc[:, :len(inputs.columns)]
output_columns = outputs.columns.drop('Drug')
results = []

for target in output_columns:
    y = merged_data[target]

    # Stratified split and synthetic oversampling for class imbalance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Initial model fitting to extract feature importance and SHAP values
    rf_initial = RandomForestClassifier(random_state=42)
    rf_initial.fit(X_train_resampled, y_train_resampled)

    # Compute SHAP values for model interpretability
    explainer = shap.TreeExplainer(rf_initial)
    shap_values = explainer.shap_values(X, check_additivity=False)

    # Feature ranking
    feature_importances = rf_initial.feature_importances_
    feature_names = X.columns
    top_5_indices = np.argsort(feature_importances)[-5:][::-1]

    print(f"\nTop 5 features for target '{target}':")
    for idx in top_5_indices:
        print(f"  Feature: {feature_names[idx]}, Importance: {feature_importances[idx]:.4f}")

    # Feature selection (top 200)
    selector = SelectFromModel(rf_initial, max_features=200, prefit=True)
    X_train_top200 = selector.transform(X_train_resampled)
    X_test_top200 = selector.transform(X_test)

    # Ensemble voting strategy with varying random seeds
    all_predictions = []
    trained_models = []
    random_states = [10, 20, 42, 57, 83]

    for seed in random_states:
        rf_model = RandomForestClassifier(random_state=seed)
        rf_model.fit(X_train_top200, y_train_resampled)

        model_filename = f"saved_models/SOC/random_forest_top200_{target}_{seed}.pkl"
        joblib.dump(rf_model, model_filename)

        trained_models.append(rf_model)
        all_predictions.append(rf_model.predict(X_test_top200))

    # Evaluate via majority voting
    majority_vote = (np.sum(all_predictions, axis=0) >= 3).astype(int)

    # Performance metrics computation
    results.append({
        'Target': target,
        'Accuracy': accuracy_score(y_test, majority_vote),
        'Precision': precision_score(y_test, majority_vote, zero_division=0),
        'Recall': recall_score(y_test, majority_vote, zero_division=0),
        'F1-Score': f1_score(y_test, majority_vote, zero_division=0),
        'y_test': y_test.values,
        'y_pred': majority_vote
    })

    print(f"Metrics for '{target}': F1-Score: {results[-1]['F1-Score']:.2f}")

# Visualization of top 10 results by F1-Score
top_10_results = sorted(results, key=lambda x: x['F1-Score'], reverse=True)[:10]

plt.figure(figsize=(20, 15))
for idx, result in enumerate(top_10_results):
    comparison_df = pd.DataFrame({'Actual': result['y_test'], 'Predicted': result['y_pred']})

    plt.subplot(2, 5, idx + 1)
    sns.heatmap(comparison_df.T, cmap='coolwarm', cbar=True, xticklabels=False, yticklabels=['Actual', 'Predicted'])
    plt.title(f"Target: {result['Target']}\nF1-Score: {result['F1-Score']:.2f}")

plt.tight_layout()
plt.show()