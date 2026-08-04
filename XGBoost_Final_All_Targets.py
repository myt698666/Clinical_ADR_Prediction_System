import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Data Ingestion
# ==========================================
print("Initializing performance benchmarking pipeline.")

slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
vip_features = slim_df.columns.tolist()

inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv", index_col='Matched Drug')
outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")

# Merge features and targets for ADR prediction
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug').dropna()

X = merged_data[vip_features].values
output_columns = outputs.columns.drop('Drug')

# ==========================================
# 2. Performance Benchmarking Routine
# ==========================================
print(f"Executing 5-Fold Stratified Cross-Validation across {len(output_columns)} targets.")

results_repository = []

for target in output_columns:
    y = merged_data[target].values

    # Skip targets with insufficient positive observations for robust validation
    if sum(y == 1) < 5:
        print(f"Skipping target '{target}': Insufficient positive observations.")
        continue

    print(f"Processing target: '{target}'")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_metrics, auprc_metrics, recall_metrics = [], [], []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Dynamic scaling for class imbalance
        pos_count = sum(y_train == 1)
        neg_count = sum(y_train == 0)
        scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

        # Gradient Boosting classifier configuration
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.9,
            scale_pos_weight=scale_weight,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)

        # Performance evaluation
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

        f1_metrics.append(f1_score(y_test, y_pred, zero_division=0))
        auprc_metrics.append(average_precision_score(y_test, y_pred_proba))
        recall_metrics.append(recall_score(y_test, y_pred, zero_division=0))

    # Aggregating cross-validation statistics
    results_repository.append({
        'SOC_Target': target,
        'Recall_Mean': np.mean(recall_metrics),
        'Recall_Std': np.std(recall_metrics),
        'F1_Score_Mean': np.mean(f1_metrics),
        'F1_Score_Std': np.std(f1_metrics),
        'AUPRC_Mean': np.mean(auprc_metrics),
        'AUPRC_Std': np.std(auprc_metrics)
    })

# ==========================================
# 3. Aggregation and Export
# ==========================================
results_df = pd.DataFrame(results_repository).sort_values(by='F1_Score_Mean', ascending=False)
output_csv_path = "ADR_Summary/Final_27_SOC_CV_Results.csv"
results_df.to_csv(output_csv_path, index=False)

print(f"\nBenchmarking finalized. Consolidated results exported to: {output_csv_path}")
print("Top 3 targets by F1-Score performance:")
print(results_df[['SOC_Target', 'Recall_Mean', 'F1_Score_Mean']].head(3).to_string(index=False))