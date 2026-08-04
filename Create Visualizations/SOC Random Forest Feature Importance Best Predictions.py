import os
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, precision_score, accuracy_score, f1_score, matthews_corrcoef,recall_score)
from sklearn.model_selection import train_test_split
# --- Create a Discrete Colormap for the Outcomes ---
from matplotlib.colors import ListedColormap, BoundaryNorm

# Create directories if they do not exist
model_dir = "Random Forests/Best Models Saved/SOC"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
fi_dir = "Feature Importance"
if not os.path.exists(fi_dir):
    os.makedirs(fi_dir)

# Load input and output files
input_file_path = "STITCH_Identifiers/drug_protein_interaction_matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)


# Merge and clean data
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')
merged_data = merged_data.dropna()

# Define feature matrix and output columns
X = merged_data.iloc[:, :-len(outputs.columns)]
output_columns = outputs.columns.drop('Drug')  # Exclude the 'Drug' column
output_columns = outputs.columns.drop('Prod')  # Exclude the 'Drug' column

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
    recall = recall_score(y_test, y_pred, zero_division=0)  # <- Added recall here
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    final_results.append({
        'Target': target,
        'ROC AUC': auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
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

# After processing all targets, save the final results to a CSV
metrics_df = pd.DataFrame(final_results)

# Define path for saving the metrics
metrics_save_path = 'Random Forests/Results/SOC_model_metrics.csv'

# Save to CSV
metrics_df.to_csv(metrics_save_path, index=False)
print(f"\nSaved model evaluation metrics to: {metrics_save_path}")

# ---- Visualization Section ----
# Set font colour to poster colour
plt.rcParams["text.color"] = "#404040"
plt.rcParams["axes.labelcolor"] = "#404040"
plt.rcParams["xtick.color"] = "#404040"
plt.rcParams["ytick.color"] = "#404040"
# Convert final_results to a DataFrame and save model metrics
results_df = pd.DataFrame(final_results)
results_csv_path = "Random Forests/Results/SOC_model_metrics.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Model metrics saved to '{results_csv_path}'.")

# Create DataFrames for predictions and actual values (using the targets that were processed)
predictions_df = pd.DataFrame(all_predictions)
actual_df = pd.DataFrame(actual_data)

# Limit the number of samples shown in the heatmap.
num_samples_to_display = 3 * len(predictions_df.columns)
predictions_df_limited = predictions_df.iloc[:num_samples_to_display, :]
actual_df_limited = actual_df.iloc[:num_samples_to_display, :]
all_drugs_count = merged_data.shape[0]
perc_drugs = (num_samples_to_display/all_drugs_count)*100

# Define a custom colormap.
custom_cmap = LinearSegmentedColormap.from_list("custom", ["#F1F2F3", "#1E84B0"], N=256)

# Create subplots for predictions vs. actual data heatmaps.
fig, axes = plt.subplots(1, 3, figsize=(10, 10), facecolor='none')

sns.heatmap(predictions_df_limited, cmap=custom_cmap, cbar=False,
            yticklabels=False, xticklabels=False, ax=axes[0], square=True)
axes[0].set_title("A) Model Predictions",fontsize=28,fontweight='bold')
#axes[0].set_xlabel("SOCs", fontsize=24,fontweight='bold')  # X-axis label
axes[0].set_ylabel(f"{perc_drugs:.0f}% Drugs", fontsize=24,fontweight='bold')  # Y-axis label

sns.heatmap(actual_df_limited, cmap=custom_cmap, cbar=False,
            yticklabels=False, xticklabels=False, ax=axes[1], square=True)
axes[1].set_title("B) Actual Data",fontsize=28, fontweight='bold')
axes[1].set_xlabel(f"{len(predictions_df.columns):.0f} SOCs", fontsize=24,fontweight='bold')  # X-axis label
#axes[1].set_ylabel("Drugs", fontsize=24,fontweight='bold')  # Y-axis label

# --- Create a Difference DataFrame ---
# We'll create a new DataFrame with the same shape as predictions_df_limited
# Each cell will be assigned a numeric code:
# 1 = True Positive (TP): prediction == 1 and actual == 1
# 2 = True Negative (TN): prediction == 0 and actual == 0
# 3 = False Positive (FP): prediction == 1 but actual == 0
# 4 = False Negative (FN): prediction == 0 but actual == 1
diff_df_numeric = predictions_df.copy()

# Loop over each cell in the full DataFrames
for i in range(diff_df_numeric.shape[0]):
    for j in range(diff_df_numeric.shape[1]):
        pred = predictions_df.iloc[i, j]
        act = actual_df.iloc[i, j]
        if pred == act:
            # Both are the same: check if positive or negative.
            diff_df_numeric.iloc[i, j] = 1 if pred == 1 else 2
        else:
            # They differ: decide between FP and FN.
            diff_df_numeric.iloc[i, j] = 3 if pred == 1 else 4

TP_full = (diff_df_numeric == 1).sum().sum()
TN_full = (diff_df_numeric == 2).sum().sum()
FP_full = (diff_df_numeric == 3).sum().sum()
FN_full = (diff_df_numeric == 4).sum().sum()
Total_full = TP_full + TN_full + FP_full + FN_full

TP_perc_full = (TP_full / Total_full) * 100
TN_perc_full = (TN_full / Total_full) * 100
FP_perc_full = (FP_full / Total_full) * 100
FN_perc_full = (FN_full / Total_full) * 100

print(f'True positive percentage: {TP_perc_full:.2f}\nTrue negative percentage: {TN_perc_full:.2f}\nFalse positive percentage: {FP_perc_full:.2f}\nFalse negative percentage: {FN_perc_full:.2f}\n')

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
# bright green for TP, pastel green for TN, Red for FP, and Orange for FN.
cmap_diff = ListedColormap(['#00B050', '#BCFFDB', '#FF6666', '#FFCCCC'])
# Define boundaries so that each code (1,2,3,4) falls in its own bin.
bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
norm = BoundaryNorm(bounds, cmap_diff.N)

ax_diff = sns.heatmap(diff_df_numeric.astype(float), cmap=cmap_diff, norm=norm,
                       cbar=True,#linewidths=0.5, linecolor='black',
                      xticklabels=False, yticklabels=False, square=True)
axes[2].set_title("Differences", fontsize=28, fontweight='bold')
#ax_diff.set_xlabel("SOCs", fontsize=24, fontweight='bold')
#ax_diff.set_ylabel("Drugs", fontsize=24, fontweight='bold')

# Adjust the colorbar to show the outcome labels.
cbar = ax_diff.collections[0].colorbar
cbar.set_ticks([1, 2, 3, 4])
cbar.set_ticklabels([f'TP:\n{TP_perc:.0f}%', f'TN:\n{TN_perc:.0f}%', f'FP:\n{FP_perc:.0f}%', f'FN:\n{FN_perc:.0f}%'])

# Modify font size and make it bold
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=22, fontweight='bold')

# Adjust the position of the colorbar labels
cbar.ax.yaxis.set_tick_params(pad=10)  # Increase padding between labels and colorbar

# Calculate the distribution percentages
distribution_labels = [f'TP:\n{TP_perc_full:.1f}%', f'TN:\n{TN_perc_full:.1f}%', f'FP:\n{FP_perc_full:.1f}%', f'FN:\n{FN_perc_full:.1f}%']
distribution_counts = [TP_full, TN_full, FP_full, FN_full]
distribution_percentages = [TP_perc_full, TN_perc_full, FP_perc_full, FN_perc_full]

# Create a figure with two subplots: one for the heatmap and one for the pie chart
fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor='none')

# Plot the heatmap (left subplot)
sns.heatmap(diff_df_numeric.astype(float), cmap=cmap_diff, norm=norm,
            cbar=True, xticklabels=False, yticklabels=False, square=True, ax=axes[0])
axes[0].set_title("C) Differences", fontsize=28, fontweight='bold')
#axes[0].set_ylabel(f"{perc_drugs:.0f}% Drugs", fontsize=24, fontweight='bold')
#axes[0].set_xlabel("SOCs", fontsize=24, fontweight='bold')

# Adjust the colorbar to show the outcome labels
cbar = axes[0].collections[0].colorbar
cbar.set_ticks([1, 2, 3, 4])
cbar.set_ticklabels([f'TP', f'TN', f'FP', f'FN'])
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=22, fontweight='bold')

# Plot the pie chart (right subplot)
axes[1].pie(distribution_counts, labels=distribution_labels, startangle=90,
            colors=['#00B050', '#BCFFDB', '#FF6666', '#FFCCCC'], textprops={'fontsize': 22, 'fontweight': 'bold'})
axes[1].set_title("D) All Data Distribution", fontsize=28, fontweight='bold')

# Adjust layout and show the figure
plt.tight_layout()
plt.show()

# Convert final_results to a DataFrame and create a long-form DataFrame for box plots.
results_df = pd.DataFrame(final_results)
metrics = ['ROC AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC']
long_df = results_df.melt(id_vars=['Target'], value_vars=metrics, var_name='Metric', value_name='Score')

plt.figure(figsize=(32, 8), facecolor='none')
ax = sns.boxplot(x='Metric', y='Score', data=long_df, showfliers=True, width=0.6, linewidth=1.2,
                 boxprops=dict(facecolor="#1E84B0", edgecolor="black"))
ax.set_facecolor("#F1F2F3")
sns.stripplot(x='Metric', y='Score', data=long_df, color='black', alpha=0.5, jitter=True)
plt.xlabel("Performance Metrics", fontsize=24,fontweight='bold')
plt.ylabel("Score",fontsize=24,fontweight='bold')
plt.title("Metrics", fontsize=32,fontweight='bold')
# Increase tick label font size
ax.tick_params(axis='both', which='major', labelsize=24, labelrotation = 90)
#plt.show()

# Plot feature importance for the top 5 models by ROC AUC.
top5 = results_df.nlargest(1, 'ROC AUC')
for target in top5['Target']:
    feature_importance_df = feature_importances[target].sort_values(by='Importance', ascending=False)
    top_features = feature_importance_df.head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
    ax.set_facecolor('#F1F2F3')
    sns.barplot(x='Importance', y='Feature', data=top_features, color="#1E84B0", ax=ax)
    
    roc_value = top5[top5['Target'] == target]['ROC AUC'].values[0]
    plt.xlabel("Feature Importance",fontsize=28,fontweight='bold')
    plt.ylabel("Target",fontsize=28,fontweight='bold')
    plt.title(f"Top 10 Feature Importances for {target}",fontsize=32,fontweight='bold')
    plt.tight_layout()
    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.show()

metrics_df = pd.DataFrame(columns=['TP', 'TN', 'FP', 'FN', 'Total', 'TP%', 'TN%', 'FP%', 'FN%'])

# Loop over each column (SOC category)
for col in predictions_df.columns:
    pred_col = predictions_df[col]
    act_col = actual_df[col]
    
    TP = ((pred_col == 1) & (act_col == 1)).sum()
    TN = ((pred_col == 0) & (act_col == 0)).sum()
    FP = ((pred_col == 1) & (act_col == 0)).sum()
    FN = ((pred_col == 0) & (act_col == 1)).sum()
    Total = TP + TN + FP + FN

    TP_perc = (TP / Total) * 100 if Total else 0
    TN_perc = (TN / Total) * 100 if Total else 0
    FP_perc = (FP / Total) * 100 if Total else 0
    FN_perc = (FN / Total) * 100 if Total else 0

    metrics_df.loc[col] = [TP, TN, FP, FN, Total, TP_perc, TN_perc, FP_perc, FN_perc]

#metrics_df.to_csv("Random Forests/Results/SOC_metrics_summary.csv")