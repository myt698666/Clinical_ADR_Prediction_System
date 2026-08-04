import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for clean log generation
warnings.filterwarnings('ignore')

print("Initializing clinical toxicity analysis and XAI framework.")

# ==========================================
# 1. Data Ingestion
# ==========================================
# Load 500-dimensional VIP feature matrix and significance targets
slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
vip_features = slim_df.columns.tolist()

inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv", index_col='Matched Drug')
outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")

# Merge features and targets
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug').dropna()
X = merged_data[vip_features]

# Target selection (System Organ Class)
target = 'Blood'
y = merged_data[target]

# ==========================================
# 2. Model Training with Optimized Hyperparameters
# ==========================================
print(f"Training XGBoost estimator for target: '{target}'.")

# Dynamic class weighting for imbalance mitigation
pos_count = sum(y == 1)
neg_count = sum(y == 0)
scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

# Initialize and fit XGBoost model with refined hyperparameters
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
xgb_model.fit(X, y)

# ==========================================
# 3. Model Interpretability Analysis (SHAP)
# ==========================================
print("Computing SHAP values for model interpretability.")

# Initialize SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)

# ==========================================
# 4. Visualization and Export
# ==========================================
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 8))

# Generate summary plot (bee swarm)
shap.summary_plot(shap_values, X, plot_type="dot", show=False)

# Export publication-quality image
output_image_path = f"SHAP_Summary_{target}.png"
plt.savefig(output_image_path, bbox_inches='tight', dpi=300)

print(f"Analysis finalized. Publication-ready chart exported to: {output_image_path}")
plt.show()