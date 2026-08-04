import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the input (top 500 protein matrix) and output (ADR significance matrix) files
input_file_path = "STITCH_Identifiers/top_500_protein_drug_interaction_matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

# Perform a left join on the 'Drug' column
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')

# Drop rows with missing output values (for all target columns)
merged_data = merged_data.dropna()

# Separate features (inputs) and target columns
X = merged_data.iloc[:, :-len(outputs.columns)]  # Inputs (all columns in `inputs`)
output_columns = outputs.columns.drop('Drug')    # Exclude the 'Drug' column
results = []

# Hyperparameter grid for tuning RandomForestClassifier
param_grid = {
    'n_estimators': [200, 350, 500],        # Number of trees in the forest
    'max_depth': [None],            # Limiting tree depth to avoid overfitting
    'min_samples_split': [2, 20],        # Avoid splitting too early on small samples
    'min_samples_leaf': [1, 10],          # Prevent creating overly specific branches
    'max_features': ['sqrt', 0.2], # Limit features per tree to prevent overfitting
}


# Train 5 models for each target column and use majority vote for the final prediction
for target in output_columns:
    y = merged_data[target]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the RandomForestClassifier
    rf_model = RandomForestClassifier(random_state=42)
    
    # Grid search for hyperparameter optimization
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    
    # Fit the model to find the best hyperparameters
    grid_search.fit(X_train, y_train)
    
    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Predict using the best model
    y_pred = best_model.predict(X_test)
    
    # Apply majority vote (not really needed here since it's a single best model)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store results for this target
    results.append({
        'Target': target,
        'Accuracy': accuracy,
        'Best_Params': best_params,
        'y_test': y_test.values,
        'y_pred': y_pred
    })
    print(f"Model trained for target '{target}' with accuracy: {accuracy:.2f}")
    print(f"Best hyperparameters: {best_params}")

# Sort results by accuracy and select the top 10 accurate models
top_10_results = sorted(results, key=lambda x: x['Accuracy'], reverse=True)[:10]

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
    plt.title(f"Heatmap for Target: {result['Target']}\nAccuracy: {result['Accuracy']:.2f}")

plt.tight_layout()
plt.show()
