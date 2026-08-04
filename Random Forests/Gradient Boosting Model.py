import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
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

# Load the input (top 500 protein matrix) and output (ADR significance matrix) files
input_file_path = "STITCH_Identifiers/drug_protein_interaction_matrix.csv"
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

from sklearn.feature_selection import SelectFromModel

results = []
for target in output_columns:
    y = merged_data[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE to balance data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Train an initial Gradient Boosting model to determine feature importance
    gb_initial = GradientBoostingClassifier(random_state=42)
    gb_initial.fit(X_train_resampled, y_train_resampled)
    
    # Get feature importances and feature names
    feature_importances = gb_initial.feature_importances_
    feature_names = X.columns
    
    # Rank and select the top 5 features
    top_5_indices = np.argsort(feature_importances)[-5:][::-1]  # Get indices of top 5 features
    top_5_features = [(feature_names[i], feature_importances[i]) for i in top_5_indices]
    
    print(f"\nTop 5 features for target '{target}':")
    for feature, importance in top_5_features:
        print(f"  Feature: {feature}, Importance: {importance:.4f}")
    
    # Select top 200 features
    selector = SelectFromModel(gb_initial, max_features=200, prefit=True)
    X_train_top200 = selector.transform(X_train_resampled)
    X_test_top200 = selector.transform(X_test)
    
    # Retrain models using only the top 200 features
    all_predictions = []
    trained_models = []
    
    for random_state in [10, 20, 42, 57, 83]:
        gb_model = GradientBoostingClassifier(random_state=random_state)
        gb_model.fit(X_train_top200, y_train_resampled)
        
        # Save model
        model_filename = f"saved_models/SOC/gradient_boosting_top200_{target}_{random_state}.pkl"
        joblib.dump(gb_model, model_filename)
        
        trained_models.append(gb_model)
        
        # Predict with the top 200 features
        y_pred = gb_model.predict(X_test_top200)
        all_predictions.append(y_pred)
    
    # Apply majority voting
    majority_vote = np.sum(all_predictions, axis=0) >= 2
    majority_vote = majority_vote.astype(int)
    
    # Evaluate model performance
    accuracy = accuracy_score(y_test, majority_vote)
    precision = precision_score(y_test, majority_vote, zero_division=0)
    recall = recall_score(y_test, majority_vote, zero_division=0)
    f1 = f1_score(y_test, majority_vote, zero_division=0)
    
    results.append({
        'Target': target,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'y_test': y_test.values,
        'y_pred': majority_vote,
        'models': trained_models
    })
    
    print(f"Model for target '{target}': Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

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
