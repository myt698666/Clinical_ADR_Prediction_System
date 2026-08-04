import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, average_precision_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Configuration and Data Loading
# ==========================================
print("Loading feature matrix and target significance matrices.")

# Load feature matrix and targets
slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
vip_features = slim_df.columns.tolist()

inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv", index_col='Matched Drug')
outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")

# Merging input features with target outcome (SOC: Blood)
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug').dropna()

X = merged_data[vip_features].values
target = 'Blood'
y = merged_data[target].values

# ==========================================
# 2. Performance Benchmarking (5-Fold Cross-Validation)
# ==========================================
print(f"Initiating 5-Fold Stratified Cross-Validation for target: '{target}'.")

# Metrics initialization
f1_scores, auprc_scores, recall_scores = [], [], []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Dynamic class weighting for imbalance mitigation
    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

    # Model definition with optimized hyperparameters
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

    # Model training
    xgb_model.fit(X_train, y_train)

    # Inference
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    # Metric computation
    f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
    auprc_scores.append(average_precision_score(y_test, y_pred_proba))
    recall_scores.append(recall_score(y_test, y_pred, zero_division=0))

    print(f"Fold {fold} | Recall: {recall_scores[-1]:.2f} | F1: {f1_scores[-1]:.2f} | AUPRC: {auprc_scores[-1]:.2f}")
    fold += 1

# ==========================================
# 3. Final Performance Evaluation Reporting
# ==========================================
print("\nBenchmark Evaluation Results:")
print("--------------------------------------------------")
print(f"Mean Recall:    {np.mean(recall_scores):.4f} (±{np.std(recall_scores):.4f})")
print(f"Mean F1-Score:  {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
print(f"Mean AUPRC:     {np.mean(auprc_scores):.4f} (±{np.std(auprc_scores):.4f})")
print("--------------------------------------------------")