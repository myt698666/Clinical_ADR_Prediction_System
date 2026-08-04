import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Data Ingestion
# ==========================================
print("Initializing hyperparameter optimization pipeline.")

# Load feature VIP list and input dataset
slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
vip_features = slim_df.columns.tolist()

inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv", index_col='Matched Drug')
outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")

# Merge features and targets
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug').dropna()

# Segregate features and target (Blood SOC classification)
X = merged_data[vip_features]
target = 'Blood'
y = merged_data[target]

print(f"Dataset successfully loaded. Target: '{target}'.")

# ==========================================
# 2. Hyperparameter Optimization Setup
# ==========================================
# Dynamic class weighting for imbalance mitigation
pos_count = sum(y == 1)
neg_count = sum(y == 0)
scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

# Define parameter search space
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

base_model = xgb.XGBClassifier(
    scale_pos_weight=scale_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

# Configuration for randomized search
f1_scorer = make_scorer(f1_score, zero_division=0)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=20,
    scoring=f1_scorer,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# ==========================================
# 3. Optimization Execution
# ==========================================
print("Initiating hyperparameter search (RandomizedSearchCV).")
random_search.fit(X, y)

# ==========================================
# 4. Result Reporting
# ==========================================
print("\nOptimization finalized. Best hyperparameter configuration:")
print("=" * 40)
print(f"Optimal Parameters: {random_search.best_params_}")
print(f"Maximum F1-Score:   {random_search.best_score_:.4f}")
print("=" * 40)