import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import os

# Ensure the output directory for models exists
os.makedirs('saved_models/SOC', exist_ok=True)

# Load the multi-modal fused feature matrix and the target significance matrix
input_file_path = "STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

# Set index to 'Matched Drug' to align with the feature matrix structure
inputs = pd.read_csv(input_file_path, index_col='Matched Drug')
outputs = pd.read_csv(output_file_path)

# Merge inputs and outputs on the drug identifier
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')
merged_data = merged_data.dropna()

# Extract feature matrix and target columns
# Exclude target columns and the 'Drug' column from the feature matrix
X = merged_data.iloc[:, :len(inputs.columns)]
output_columns = outputs.columns.drop('Drug')
results = []

# Define parameter grid for hyperparameter optimization
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 15, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Iterate through each System Organ Class (SOC) target
for target in output_columns:
    y = merged_data[target]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Address class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=X.columns)

    # Initialize RandomForest and GridSearch
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1, scoring='f1')
    grid_search.fit(X_train_resampled_df, y_train_resampled)

    best_rf_model = grid_search.best_estimator_
    print(f"Target: {target} | Best Hyperparameters: {grid_search.best_params_}")

    # Feature selection based on importance threshold
    selector = SelectFromModel(best_rf_model, threshold="mean", max_features=200)
    selector.fit(X_train_resampled_df, y_train_resampled)

    selected_features = X_train_resampled_df.columns[selector.get_support()]
    print(f"Target: {target} | Number of selected features: {len(selected_features)}")

    # Save model artifact
    model_filename = f"saved_models/SOC/random_forest_best_{target}.pkl"
    joblib.dump(best_rf_model, model_filename)

    # Transform data and perform final evaluation
    X_train_selected = selector.transform(X_train_resampled_df)
    X_test_selected = selector.transform(X_test)
    best_rf_model.fit(X_train_selected, y_train_resampled)
    y_pred = best_rf_model.predict(X_test_selected)

    # Calculate performance metrics
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    results.append({
        'Target': target,
        'F1-Score': f1,
        'Precision': precision,
        'Recall': recall,
        'y_test': y_test.values,
        'y_pred': y_pred
    })

    print(f"Target: {target} | F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

# Visualize top 10 models performance
top_10_results = sorted(results, key=lambda x: x['F1-Score'], reverse=True)[:10]

plt.figure(figsize=(20, 15))
for idx, result in enumerate(top_10_results):
    comparison_df = pd.DataFrame({'Actual': result['y_test'], 'Predicted': result['y_pred']})

    plt.subplot(2, 5, idx + 1)
    sns.heatmap(comparison_df.T, cmap='coolwarm', cbar=False, xticklabels=False, yticklabels=['Actual', 'Predicted'])
    plt.title(f"Target: {result['Target']}\nF1: {result['F1-Score']:.2f}")

plt.tight_layout()
plt.show()