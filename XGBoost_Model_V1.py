import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
import warnings

# Suppress version warnings for cleaner log outputs
warnings.filterwarnings('ignore')

# ==========================================
# 1. Data Ingestion and Preprocessing
# ==========================================
print("Initializing feature matrix and target significance matrices.")

# Load feature VIP list (500 most informative features)
slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
vip_features = slim_df.columns.tolist()

# Load primary dataset and labels
input_file_path = "STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col='Matched Drug')
outputs = pd.read_csv(output_file_path)

# Alignment of features and target outcomes
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug').dropna()

# Refine feature space to the 500-dimensional VIP matrix
X = merged_data[vip_features]
output_columns = outputs.columns.drop('Drug')

print(f"Feature space initialized: {X.shape[0]} samples, {X.shape[1]} dimensions.")
print("Initiating XGBoost classification pipeline with automated class-balancing.")

# ==========================================
# 2. Performance Benchmarking Routine
# ==========================================
for target in output_columns:
    y = merged_data[target]

    # Partitioning the dataset (80% training, 20% validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dynamic class weighting for imbalance mitigation
    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

    # Model configuration
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )

    # Training and inference
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metric evaluation
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'AUPRC': average_precision_score(y_test, y_pred_proba)
    }

    # Logging results
    log_string = f"Target: '{target}' | " + " | ".join([f"{k}={v:.2f}" for k, v in metrics.items()])
    print(log_string)

print("\nPerformance benchmarking completed.")