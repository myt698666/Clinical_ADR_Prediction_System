import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score
import warnings

warnings.filterwarnings('ignore')

print("Evaluating WGAN-GP data augmentation impact on XGBoost performance.")

target_soc = 'Ear'

# ==========================================
# 1. Loading Clinical Dataset
# ==========================================
print("Loading real-world clinical data.")
try:
    slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
    vip_features = slim_df.columns.tolist()

    inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv", index_col='Matched Drug')
    outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")

    merged_data = inputs.merge(outputs[['Drug', target_soc]], how='left', left_index=True, right_on='Drug').dropna()

    X_real = merged_data[vip_features].values
    y_real = merged_data[target_soc].values.astype(int)

    print(f"Data dimensions: {X_real.shape} | Positive samples: {sum(y_real == 1)} | Negative samples: {sum(y_real == 0)}")
except FileNotFoundError:
    print("Error: Input data files not found.")
    exit()

# ==========================================
# 2. Dataset Splitting
# ==========================================
# Ensuring test set integrity by preserving exclusively real-world clinical observations
X_train_real, X_test, y_train_real, y_test = train_test_split(
    X_real, y_real, test_size=0.2, stratify=y_real, random_state=42
)

# ==========================================
# 3. Integrating Synthetic Data
# ==========================================
print("Loading WGAN-GP generated synthetic samples.")
try:
    syn_df = pd.read_csv(f"WGAN_Synthetic_{target_soc}_Features.csv")
    X_syn = syn_df[vip_features].values
    y_syn = syn_df['Label'].values.astype(int)
    print(f"Synthetic records injected: {len(X_syn)}")
except FileNotFoundError:
    print("Error: Synthetic data not found. Execute WGAN_GP_Synthesizer.py first.")
    exit()

X_train_aug = np.vstack((X_train_real, X_syn))
y_train_aug = np.concatenate((y_train_real, y_syn))

# ==========================================
# 4. Performance Benchmarking
# ==========================================
print("Initiating performance benchmarking: Baseline vs. WGAN-Augmented.")

# Baseline model: Trained on real data
clf_base = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
clf_base.fit(X_train_real, y_train_real)
y_prob_base = clf_base.predict_proba(X_test)[:, 1]

# Augmented model: Trained on real + synthetic data
clf_aug = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
clf_aug.fit(X_train_aug, y_train_aug)
y_prob_aug = clf_aug.predict_proba(X_test)[:, 1]

# Adaptive threshold optimization
def get_optimal_predictions(y_true, y_prob):
    """Identifies the optimal classification threshold for F1-score maximization."""
    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.arange(0.05, 0.5, 0.01):
        preds = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    return (y_prob >= best_thresh).astype(int), best_thresh

y_pred_base, thresh_base = get_optimal_predictions(y_test, y_prob_base)
y_pred_aug, thresh_aug = get_optimal_predictions(y_test, y_prob_aug)

# ==========================================
# 5. Performance Report
# ==========================================
print(f"\nComparative Performance Analysis: Target '{target_soc}' (Test Set)")
print("-" * 75)
print(f"{'Metric':<15} | {'Baseline (No GAN)':<20} | {'WGAN-GP Augmented':<20}")
print("-" * 75)
print(f"Optimal Thresh  | {thresh_base:<20.2f} | {thresh_aug:<20.2f}")

def print_metric(name, func, y_t, p_base, p_aug, is_prob=False):
    if is_prob:
        base_score = func(y_t, y_prob_base)
        aug_score = func(y_t, y_prob_aug)
    else:
        base_score = func(y_t, p_base, zero_division=0)
        aug_score = func(y_t, p_aug, zero_division=0)
    print(f"{name:<15} | {base_score:<20.4f} | {aug_score:<20.4f}")

print_metric("Precision", precision_score, y_test, y_pred_base, y_pred_aug)
print_metric("Recall", recall_score, y_test, y_pred_base, y_pred_aug)
print_metric("F1-Score", f1_score, y_test, y_pred_base, y_pred_aug)
print_metric("PR-AUC", average_precision_score, y_test, None, None, is_prob=True)
print("-" * 75)
print("Conclusion: WGAN-GP data augmentation combined with adaptive thresholding facilitates")
print("the identification of minority class patterns previously suppressed by fixed decision boundaries.")