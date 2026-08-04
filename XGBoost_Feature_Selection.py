import pandas as pd
import xgboost as xgb
import numpy as np
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

print("🚀 [1/4] 正在加载 2548 维原始超级矩阵...")
input_file_path = "STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col='Matched Drug')
outputs = pd.read_csv(output_file_path)

merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')
merged_data = merged_data.dropna()

X = merged_data.iloc[:, :-len(outputs.columns)]
output_columns = outputs.columns.drop('Drug')

# 准备一个空数组，用来记录所有特征的“累计战功”
global_importances = np.zeros(X.shape[1])

print(f"🔍 [2/4] 正在让 XGBoost 评委对所有特征进行全网扫描 (耗时约1-2分钟)...")

# 让 XGBoost 重新跑一遍 27 个副作用，收集特征重要性
for target in output_columns:
    y = merged_data[target]

    # 计算惩罚权重
    pos_count = sum(y == 1)
    neg_count = sum(y == 0)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1
    )

    # 这次我们用所有数据来训练，以获得最准确的全局特征重要性
    xgb_model.fit(X, y)

    # 将这个副作用分类下特征的战功，累加到总榜单里
    global_importances += xgb_model.feature_importances_

# 制作战功排行榜
feat_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Total_Importance': global_importances
}).sort_values(by='Total_Importance', ascending=False)

# 💡 核心操作：我们只保留排名前 500 的超级特征！(你可以随时修改这个数字)
TOP_N = 500
top_features = feat_imp_df['Feature'].head(TOP_N).tolist()

print(f"\n🏆 [3/4] 战功榜揭晓！排名前 5 的终极特征是：")
for i, row in feat_imp_df.head(5).iterrows():
    print(f"   - {row['Feature']} (战功值: {row['Total_Importance']:.4f})")

print(f"\n✂️ [4/4] 正在砍掉 {X.shape[1] - TOP_N} 个摸鱼特征，生成瘦身版矩阵...")

# 提取排名前 500 的特征列，生成新矩阵
X_slim = X[top_features]

# 保存这个极其珍贵的提纯版矩阵
slim_output_path = "STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv"
X_slim.to_csv(slim_output_path)

print(f"💾 搞定！【500维黄金矩阵】已成功保存至: {slim_output_path}")
print("   (明天汇报又多了一个绝佳的吹牛资本！)")