import os
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, precision_score, accuracy_score, f1_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split

# Create directories if they do not exist
model_dir = "Random Forests/Best Models Saved"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
fi_dir = "Feature Importance"
if not os.path.exists(fi_dir):
    os.makedirs(fi_dir)

# Load input and output files
input_file_path = "STITCH_Identifiers/drug_protein_interaction_matrix.csv"
output_file_path = "ADR_Summary/HLGT_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

# Merge and clean data
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')
merged_data = merged_data.dropna()

# Define feature matrix and output columns
X = merged_data.iloc[:, :-len(outputs.columns)]
output_columns = outputs.columns.drop('Drug')  # Exclude the 'Drug' column

# Dictionaries for storing results, predictions, and feature importances
final_results = []
feature_importances = {}
all_predictions = {}   # Will store predictions for each target
actual_data = {}       # Will store actual y values for each target

# Loop over each target
for target in output_columns:
    print(f"\nProcessing target: {target}")
    
    # Define file paths for best parameters and saved model
    params_file = f"Random Forests/Best Models/{target}_best_params.json"
    model_file = os.path.join(model_dir, f"{target}_rf_model.joblib")
    
    # Split the data (we use the same split for both training and testing, regardless of training or loading)
    y = merged_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # If a saved model exists, load it. Otherwise, train a new model.
    if os.path.exists(model_file):
        print(f"Loading saved model for '{target}' from {model_file}.")
        final_rf_model = joblib.load(model_file)
    else:
        # If best parameter file does not exist, skip target.
        if not os.path.exists(params_file):
            print(f"Best params file for target '{target}' not found. Skipping...")
            continue
        try:
            with open(params_file, "r") as f:
                best_params = json.load(f)
        except FileNotFoundError:
            print(f"Best params file for target '{target}' not found (caught in try/except). Skipping...")
            continue
        
        params = best_params.get(target, {})
        
        # Handle SMOTE: check the positive count in the training data.
        positive_count = (y_train == 1).sum()
        if positive_count < 2:
            print(f"Target '{target}': too few positive data points in training set ({positive_count}). Skipping...")
            continue
        elif positive_count < 5:
            try:
                k_neighbors_val = min(5, np.min(y_train.value_counts()) - 1)
                smote = SMOTE(k_neighbors=k_neighbors_val, random_state=42)
            except Exception as e:
                print(f"Target '{target}': error determining k_neighbors for SMOTE - {e}. Skipping...")
                continue
        else:
            smote = SMOTE(random_state=42)
        
        try:
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        except ValueError as e:
            print(f"Target '{target}': Skipping due to SMOTE error - {e}")
            continue
        
        # Create and train the RandomForest model using the best parameters.
        final_rf_model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 54),
            min_samples_split=params.get('min_samples_split', 6),
            min_samples_leaf=params.get('min_samples_leaf', 18),
            max_features=params.get('max_features', 'sqrt'),
            class_weight=params.get('class_weight', 'balanced'),
            random_state=42
        )
        final_rf_model.fit(X_train_res, y_train_res)
        
        # Save the newly trained model to disk for future reuse.
        joblib.dump(final_rf_model, model_file)
        print(f"Saved model for '{target}' to {model_file}.")

    # Use the model for predictions on the test set.
    try:
        y_pred_prob = final_rf_model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"Error predicting probabilities for '{target}': {e}. Skipping...")
        continue
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Evaluate model performance.
    try:
        auc = roc_auc_score(y_test, y_pred_prob)
    except Exception:
        auc = 0.0
    precision = precision_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    final_results.append({
        'Target': target,
        'ROC AUC': auc,
        'Precision': precision,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'MCC': mcc,
        'y_test': y_test.values,
        'y_pred': y_pred
    })
    print(f"Final model for '{target}' - AUC: {auc:.2f}, Precision: {precision:.2f}, Accuracy: {accuracy:.2f}, F1: {f1:.2f}, MCC: {mcc:.2f}")
    
    # Compute and save feature importances.
    importances = final_rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_importances[target] = feature_importance_df
    fi_save_path = os.path.join(fi_dir, f"feature_importance_{target}.csv")
    feature_importance_df.to_csv(fi_save_path, index=False)
    print(f"Feature importance for '{target}' saved to {fi_save_path}.")
    
    # Store predictions and actual values for later visualization.
    all_predictions[target] = y_pred
    actual_data[target] = y_test.values

# ---- Visualization Section ----

# Create DataFrames for predictions and actual values (using the targets that were processed)
predictions_df = pd.DataFrame(all_predictions)
actual_df = pd.DataFrame(actual_data)
results_df = pd.DataFrame(final_results)

# 1. Select the top 100 models based on ROC AUC.
# Note: We sort results_df by the 'ROC AUC' column and then extract the 'Target' names.
top_100_targets = results_df.sort_values(by='ROC AUC', ascending=False)['Target'].iloc[:100].tolist()

# 2. Reorder these top 100 targets alphabetically.
top_100_targets = sorted(top_100_targets)

# 3. Subset the predictions and actual DataFrames to only these top 100 targets (columns).
# Since predictions_df and actual_df have targets as columns, we use these as our new column order.
predictions_df_ordered = predictions_df[top_100_targets]
actual_df_ordered = actual_df[top_100_targets]

# Optionally, if you wish to limit the number of rows displayed (for example, to 3 * number of targets):
num_rows_to_display = 3 * len(top_100_targets)
predictions_df_limited = predictions_df_ordered.iloc[:num_rows_to_display, :]
actual_df_limited = actual_df_ordered.iloc[:num_rows_to_display, :]
all_drugs_count = merged_data.shape[0]
perc_drugs = (num_rows_to_display/all_drugs_count)*100

# Save model metrics
results_csv_path = "Random Forests/Results/HLGT_model_metrics.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Model metrics saved to '{results_csv_path}'.")

# Define a custom colormap.
custom_cmap = LinearSegmentedColormap.from_list("custom", ["#F1F2F3", "#1E84B0"], N=256)

# Create subplots for predictions vs. actual data heatmaps.
fig, axes = plt.subplots(1, 3, figsize=(24, 10), facecolor='none')

sns.heatmap(predictions_df_limited, cmap=custom_cmap, cbar=False,
            yticklabels=False, xticklabels=False, ax=axes[0], square=True,)#, linewidths=0.1,linecolor='black')
axes[0].set_title("Model Predictions", fontsize=28, fontweight='bold')
#axes[0].set_xlabel("HLGTs", fontsize=24, fontweight='bold')
axes[0].set_ylabel(f"{perc_drugs:.0f}% Drugs", fontsize=24, fontweight='bold')

sns.heatmap(actual_df_limited, cmap=custom_cmap, cbar=False,
            yticklabels=False, xticklabels=False, ax=axes[1], square=True)#, linewidths=0.1,linecolor='black')
axes[1].set_title("Actual Data", fontsize=28, fontweight='bold')
axes[1].set_xlabel(f"{len(predictions_df.columns):.0f} HLGTs", fontsize=24, fontweight='bold')
#axes[1].set_ylabel("Drugs", fontsize=24, fontweight='bold')

# --- Create a Difference DataFrame ---
# We'll create a new DataFrame with the same shape as predictions_df_limited
# Each cell will be assigned a numeric code:
# 1 = True Positive (TP): prediction == 1 and actual == 1
# 2 = True Negative (TN): prediction == 0 and actual == 0
# 3 = False Positive (FP): prediction == 1 but actual == 0
# 4 = False Negative (FN): prediction == 0 but actual == 1

diff_df_numeric = predictions_df_limited.copy()

# Loop over each cell (you could also use vectorized operations if desired)
for i in range(diff_df_numeric.shape[0]):
    for j in range(diff_df_numeric.shape[1]):
        pred = predictions_df_limited.iloc[i, j]
        act = actual_df_limited.iloc[i, j]
        if pred == act:
            # Both are the same: check if positive or negative.
            diff_df_numeric.iloc[i, j] = 1 if pred == 1 else 2
        else:
            # They differ: decide between FP and FN.
            diff_df_numeric.iloc[i, j] = 3 if pred == 1 else 4

# --- Create a Discrete Colormap for the Outcomes ---
from matplotlib.colors import ListedColormap, BoundaryNorm

TP = (diff_df_numeric==1).sum().sum()
TN = (diff_df_numeric==2).sum().sum()
FP = (diff_df_numeric==3).sum().sum()
FN = (diff_df_numeric==4).sum().sum()
Total = TP+TN+FP+FN
TP_perc = (TP/Total)*100
TN_perc = (TN/Total)*100
FP_perc = (FP/Total)*100
FN_perc = (FN/Total)*100

# Define colors for each outcome:
# Green for TP, Blue for TN, Red for FP, and Orange for FN.
cmap_diff = ListedColormap(['#00B050', '#BCFFDB', '#FF6666', '#FFCCCC'])
# Define boundaries so that each code (1,2,3,4) falls in its own bin.
bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
norm = BoundaryNorm(bounds, cmap_diff.N)

ax_diff = sns.heatmap(diff_df_numeric.astype(float), cmap=cmap_diff, norm=norm,
                       cbar=True,#linewidths=0.5, linecolor='black',
                      xticklabels=False, yticklabels=False, square=True)
ax_diff.set_title("Difference Chart", fontsize=28, fontweight='bold')
#ax_diff.set_xlabel("HLGTs", fontsize=24, fontweight='bold')
#ax_diff.set_ylabel("Drugs", fontsize=24, fontweight='bold')

# Adjust the colorbar to show the outcome labels.
cbar = ax_diff.collections[0].colorbar
cbar.set_ticks([1, 2, 3, 4])
cbar.set_ticklabels([f'TP:\n{TP_perc:.0f}%', f'TN:\n{TN_perc:.0f}%', f'FP:\n{FP_perc:.0f}%', f'FN:\n{FN_perc:.0f}%'])
# Modify font size and make it bold
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=24, fontweight='bold')

plt.tight_layout()
plt.show()


# Convert final_results to a DataFrame and create a long-form DataFrame for box plots.
results_df = pd.DataFrame(final_results)
metrics = ['ROC AUC', 'Precision', 'Accuracy', 'F1 Score', 'MCC']
long_df = results_df.melt(id_vars=['Target'], value_vars=metrics, var_name='Metric', value_name='Score')

plt.figure(figsize=(32, 8), facecolor='none')
ax = sns.boxplot(x='Metric', y='Score', data=long_df, showfliers=True, width=0.6, linewidth=1.2,
                 boxprops=dict(facecolor="#1E84B0", edgecolor="black"))
ax.set_facecolor("#F1F2F3")
sns.stripplot(x='Metric', y='Score', data=long_df, color='black', alpha=0.5, jitter=True)
plt.xlabel("Performance Metrics", fontsize=24,fontweight='bold')
plt.ylabel("Score",fontsize=24,fontweight='bold')
plt.title("Metrics", fontsize=32, fontweight='bold')
# Increase tick label font size
ax.tick_params(axis='both', which='major', labelsize=24)
plt.show()

# Exclude models with ROC AUC equal to 1 for the feature importance plots.
results_df_filtered = results_df[results_df['ROC AUC'] < 0.998]

# Check if we have any models left after filtering.
if results_df_filtered.empty:
    print("No models with ROC AUC < 1.0 were found for feature importance visualization.")
else:
    # Get the top 5 models by ROC AUC from the filtered results.
    top5 = results_df_filtered.nlargest(5, 'ROC AUC')
    
    # Loop over the targets of the top 5 models and plot their feature importance.
    for target in top5['Target']:
        # Retrieve and sort the feature importances for the current target.
        feature_importance_df = feature_importances[target].sort_values(by='Importance', ascending=False)
        
        # Select the top 10 features for clarity.
        top_features = feature_importance_df.head(10)
        
        # Create a barplot for the feature importances.
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
        ax.set_facecolor('#F1F2F3')
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=top_features, 
            color="#1E84B0", 
            ax=ax
        )
        
        # Add labels and a title that includes the ROC AUC value.
        roc_value = top5[top5['Target'] == target]['ROC AUC'].values[0]
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title(f"Top 10 Feature Importances for {target}\n(ROC AUC: {roc_value:.2f})")
        plt.tight_layout()
        plt.show()