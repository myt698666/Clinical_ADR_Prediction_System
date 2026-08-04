import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("🚀 [1/4] 正在加载 500 维黄金 VIP 燃料...")
slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
vip_features = slim_df.columns.tolist()

inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv", index_col='Matched Drug')
outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug').dropna()

X = merged_data[vip_features]
target = 'Blood'  # 我们依然先拿血液系统开刀
y = merged_data[target]

print(f"🎯 [2/4] 正在使用刚刚找到的【最强冠军参数】训练 '{target}' 模型...")

# 自动计算不平衡权重
pos_count = sum(y == 1)
neg_count = sum(y == 0)
scale_weight = neg_count / pos_count if pos_count > 0 else 1

# 💡 这里填入的就是你上一局跑出来的最佳参数！
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

print("🔍 [3/4] 模型训练完毕！正在请 SHAP 大师进行开箱解剖 (可能需要十几秒)...")

# 初始化 SHAP 解释器并计算 SHAP 值
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)

print("🎨 [4/4] 正在绘制 SHAP 蜂巢图 (Bee Swarm Plot)...")

# 解决画图时可能出现的中文/负号显示问题（学术图表防踩坑）
plt.rcParams['axes.unicode_minus'] = False

# 画图并展示
plt.figure(figsize=(10, 8)) # 设定画布大小
shap.summary_plot(shap_values, X, plot_type="dot", show=False)

# 保存这张极其珍贵的学术图表
output_image_path = f"SHAP_Summary_{target}.png"
plt.savefig(output_image_path, bbox_inches='tight', dpi=300)
print(f"\n🎉 大功告成！顶级学术图表已保存至当前目录: {output_image_path}")

# 在你的 PyCharm 里直接弹出来给你看！
plt.show()