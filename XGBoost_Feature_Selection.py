import pandas as pd
import numpy as np
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Configuration and Data Ingestion
# ==========================================
print("Initializing feature selection engine.")

input_file_path = "STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col='Matched Drug')
outputs = pd.read_csv(output_file_path)

# Merge feature matrix and significance targets
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug').dropna()

# Segregate features and targets
X = merged_data.iloc[:, :len(inputs.columns)]
output_columns = outputs.columns.drop('Drug')

# ==========================================
# 2. Global Feature Importance Aggregation
# ==========================================
print("Executing cumulative feature importance analysis via XGBoost Gradient Boosting.")

# Aggregate importance scores across all clinical ADR targets
global_importances = np.zeros(X.shape[1])

for target in output_columns:
    y = merged_data[target]

    # Dynamic class weighting for imbalance mitigation
    pos_count = sum(y == 1)
    neg_count = sum(y == 0)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

    # Fit Gradient Boosting model to estimate feature relevance
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    # Cumulative importance accumulation
    global_importances += model.feature_importances_

# ==========================================
# 3. Dimensionality Reduction and Refinement
# ==========================================
# Generate ranking of feature importances
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Total_Importance': global_importances
}).sort_values(by='Total_Importance', ascending=False)

# Select top-N informative features (N=500)
TOP_N = 500
top_features = importance_df['Feature'].head(TOP_N).tolist()

print("\nFeature ranking summary (Top 5):")
for _, row in importance_df.head(5).iterrows():
    print(f"  Feature: {row['Feature']} | Importance Score: {row['Total_Importance']:.4f}")

# Refine feature matrix
X_refined = X[top_features]

# Persist the refined matrix to disk
slim_output_path = "STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv"
X_refined.to_csv(slim_output_path)

print(f"\nFeature dimensionality reduction completed.")
print(f"Original dimensionality: {X.shape[1]}")
print(f"Refined dimensionality: {X_refined.shape[1]}")
print(f"Refined matrix persisted to: {slim_output_path}")