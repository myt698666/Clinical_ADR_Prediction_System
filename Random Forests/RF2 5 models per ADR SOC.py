import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score


# Create the saved models folder if it doesn't exist
os.makedirs('saved_models/SOC', exist_ok=True)

# --- 外科手术替换开始：注入咱们的 2548 维多模态超级矩阵 ---
input_file_path = "STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

# 读取超级矩阵，并把咱们的 'Matched Drug' 列设为索引，完美对接原作者的代码逻辑
inputs = pd.read_csv(input_file_path, index_col='Matched Drug')
outputs = pd.read_csv(output_file_path)
# --- 外科手术替换结束 ---

# Perform a left join on the 'Drug' column
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')

# Drop rows with missing output values (for all target columns)
merged_data = merged_data.dropna()

# Separate features (inputs) and target columns
X = merged_data.iloc[:, :-len(outputs.columns)]  # Inputs (all columns in `inputs`)
output_columns = outputs.columns.drop('Drug')    # Exclude the 'Drug' column
results = []

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [None, 15, 30],  # Maximum depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],    # Minimum samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for the best split
}

# Modify the loop to include GridSearchCV for hyperparameter optimization
for target in output_columns:
    y = merged_data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to balance data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Convert X_train_resampled to DataFrame with feature names
    X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=X.columns)

    # Initialize a RandomForestClassifier (no hyperparameters for now)
    rf_model = RandomForestClassifier(random_state=42)

    # Set up GridSearchCV to search for the best hyperparameters with F1-score as the scoring metric
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='f1')

    # Fit GridSearchCV to find the best hyperparameters
    grid_search.fit(X_train_resampled_df, y_train_resampled)

    # Get the best model from GridSearchCV
    best_rf_model = grid_search.best_estimator_

    # Print the best hyperparameters found by GridSearchCV
    print(f"Best hyperparameters for target '{target}': {grid_search.best_params_}")

    # Use SelectFromModel to select the most important features
    selector = SelectFromModel(best_rf_model, threshold="mean", max_features=200)  # Select top 200 features
    selector.fit(X_train_resampled_df, y_train_resampled)
    
    # Get the selected features
    selected_features = X_train_resampled_df.columns[selector.get_support()]

    # Print the selected features
    print(f"Top selected features for target '{target}': {selected_features.tolist()}")

    # Save the best model
    model_filename = f"saved_models/SOC/random_forest_best_{target}_best.pkl"
    joblib.dump(best_rf_model, model_filename)

    # Transform the train and test sets to include only the selected features
    X_train_selected = selector.transform(X_train_resampled_df)
    X_test_selected = selector.transform(X_test)

    # Train the best model on the selected features
    best_rf_model.fit(X_train_selected, y_train_resampled)

    # Predict using the best model
    y_pred = best_rf_model.predict(X_test_selected)

    # Evaluate model performance using F1-score, precision, and recall
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # Store results
    results.append({
        'Target': target,
        'F1-Score': f1,
        'Precision': precision,
        'Recall': recall,
        'y_test': y_test.values,
        'y_pred': y_pred,
        'best_model': best_rf_model
    })

    # Output the performance metrics for each target
    print(f"Model for target '{target}': F1-Score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

# Sort results by accuracy and select the top 10 accurate models
top_10_results = sorted(results, key=lambda x: x['F1-Score'], reverse=True)[:10]


# Predict with the top 10 models and visualize the validation results (y_pred vs y_test)
plt.figure(figsize=(20, 15))
for idx, result in enumerate(top_10_results):
    # Compare the predicted values to the actual ones (y_test) for this target
    comparison_df = pd.DataFrame({
        'Actual': result['y_test'],
        'Predicted': result['y_pred']
    })
    
    plt.subplot(2, 5, idx + 1)  # 2 rows, 5 columns for 10 plots
    sns.heatmap(
        comparison_df.T, 
        cmap='coolwarm', 
        cbar=True, 
        xticklabels=False, 
        yticklabels=['Actual', 'Predicted']
    )
    plt.title(f"Heatmap for Target: {result['Target']}\nF1-score: {result['F1-Score']:.2f}")

plt.tight_layout()
plt.show()
