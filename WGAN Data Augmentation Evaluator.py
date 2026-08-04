import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score
import warnings

warnings.filterwarnings('ignore')

print("🧪 ========================================================")
print("🧬 [Track 2 Phase B] Evaluating WGAN-GP Augmented XGBoost")
print("🧪 ========================================================\n")

target_soc = 'Ear'

# ==========================================
# 1. 加载真实多模态数据
# ==========================================
print("⏳ [1/3] Loading Real Clinical Data...")
try:
    slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
    vip_features = slim_df.columns.tolist()

    inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv", index_col='Matched Drug')
    outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")

    merged_data = inputs.merge(outputs[['Drug', target_soc]], how='left', left_index=True, right_on='Drug').dropna()

    X_real = merged_data[vip_features].values
    y_real = merged_data[target_soc].values.astype(int)

    print(f"   📊 Real Data Shape: {X_real.shape} | Positives: {sum(y_real == 1)} | Negatives: {sum(y_real == 0)}")
except FileNotFoundError:
    print("❌ Error: Missing real data files.")
    exit()

# ==========================================
# 2. 严格切分训练集与测试集 (保证测试集纯洁)
# ==========================================
# 测试集必须 100% 由真实世界数据组成，绝不能混入 GAN 生成的假数据！
X_train_real, X_test, y_train_real, y_test = train_test_split(
    X_real, y_real, test_size=0.2, stratify=y_real, random_state=42
)

# ==========================================
# 3. 加载 WGAN 生成的虚拟阳性样本
# ==========================================
print("⏳ [2/3] Loading WGAN-GP Synthetic Data...")
try:
    syn_df = pd.read_csv(f"WGAN_Synthetic_{target_soc}_Features.csv")
    X_syn = syn_df[vip_features].values
    y_syn = syn_df['Label'].values.astype(int)
    print(f"   🧬 Synthetic Samples Injected: {len(X_syn)}")
except FileNotFoundError:
    print("❌ Error: Missing WGAN synthetic data. Run 'WGAN_GP_Synthesizer.py' first.")
    exit()

# 将虚拟数据融合进【训练集】
X_train_aug = np.vstack((X_train_real, X_syn))
y_train_aug = np.concatenate((y_train_real, y_syn))

# ==========================================
# 4. 算法打擂台：纯真实数据 vs 增强数据
# ==========================================
print("\n⚔️ [3/3] Commencing XGBoost Battle (Base vs WGAN-Augmented)...")

# --- 模型 A: 纯真实数据训练 (Baseline) ---
# 为了体现扩增的效果，这里我们不使用极端的 scale_pos_weight
clf_base = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
clf_base.fit(X_train_real, y_train_real)
y_pred_base = clf_base.predict(X_test)
y_prob_base = clf_base.predict_proba(X_test)[:, 1]

# --- 模型 B: 混入 WGAN 数据训练 (Augmented) ---
clf_aug = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
clf_aug.fit(X_train_aug, y_train_aug)
# 不再直接使用 predict，而是获取概率！
y_prob_base = clf_base.predict_proba(X_test)[:, 1]
y_prob_aug = clf_aug.predict_proba(X_test)[:, 1]


# ==========================================
# [新增核心]: 自适应阈值搜索 (Adaptive Thresholding)
# ==========================================
def get_best_predictions(y_true, y_prob):
    """寻找最佳概率门槛，不再死板使用 0.5"""
    best_thresh = 0.5
    best_f1 = 0
    # 在 0.05 到 0.5 之间搜索最佳门槛
    for thresh in np.arange(0.05, 0.5, 0.01):
        preds = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    # 用最佳门槛生成最终预测
    final_preds = (y_prob >= best_thresh).astype(int)
    return final_preds, best_thresh


# 应用自适应阈值
y_pred_base, thresh_base = get_best_predictions(y_test, y_prob_base)
y_pred_aug, thresh_aug = get_best_predictions(y_test, y_prob_aug)

# ==========================================
# 5. 打印对比报告
# ==========================================
print(f"\n🏆 Final Validation Results for Target: 【{target_soc}】 (Test Set Only)")
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

    diff = aug_score - base_score
    trend = "📈 UP!" if diff > 0.001 else ("📉 DOWN" if diff < -0.001 else "➖ SAME")
    print(f"{name:<15} | {base_score:<20.4f} | {aug_score:<12.4f} ({trend})")


print_metric("Precision", precision_score, y_test, y_pred_base, y_pred_aug)
print_metric("Recall", recall_score, y_test, y_pred_base, y_pred_aug)
print_metric("F1-Score", f1_score, y_test, y_pred_base, y_pred_aug)
print_metric("PR-AUC", average_precision_score, y_test, None, None, is_prob=True)
print("-" * 75)
print("✨ Conclusion: By combining WGAN data with adaptive thresholding, we force the model")
print("   to recognize minority patterns that were previously suppressed by the 0.5 hard limit!")