import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import f1_score, make_scorer
import warnings

warnings.filterwarnings('ignore')

print("🚀 [1/4] 正在加载 500 维黄金 VIP 燃料...")

# 1. 读取 VIP 黄金名单
slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
vip_features = slim_df.columns.tolist()

# 2. 读取带名字的完整数据并对齐
inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv", index_col='Matched Drug')
outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug').dropna()

# 提取特征
X = merged_data[vip_features]

# 💡 注意：为了节省时间，我们这次只拿 'Blood' (血液系统) 这一个目标来专门调优！
target = 'Blood'
y = merged_data[target]

print(f"🎯 [2/4] 已锁定目标 '{target}'，开始配置自动调优网格...")

# 计算类别权重 (解决不平衡)
pos_count = sum(y == 1)
neg_count = sum(y == 0)
scale_weight = neg_count / pos_count if pos_count > 0 else 1

# 划定我们允许电脑去尝试的“参数范围” (Hyperparameter Grid)
param_grid = {
    'n_estimators': [100, 200, 300, 500],       # 树的数量
    'max_depth': [3, 5, 7, 9],                  # 树的深度 (太深容易过拟合)
    'learning_rate': [0.01, 0.05, 0.1, 0.2],    # 学习率 (步子迈多大)
    'subsample': [0.7, 0.8, 0.9, 1.0],          # 每次随机抽取多少样本
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]    # 每次随机抽取多少特征
}

# 建立基础引擎
base_model = xgb.XGBClassifier(scale_pos_weight=scale_weight, random_state=42, n_jobs=-1)

# 告诉机器我们最看重的指标是 F1 分数
f1_scorer = make_scorer(f1_score, zero_division=0)

# 启动自动搜索器 (尝试 20 种随机组合，使用 3 折交叉验证)
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=20,               # 随机抽查 20 种不同的配置
    scoring=f1_scorer,       # 唯一目标：把 F1 搞上去！
    cv=3,                    # 每次测试考 3 遍取平均
    verbose=2,               # 打印寻找的过程
    random_state=42,
    n_jobs=-1                # 全核火力全开
)

print("⏳ [3/4] 电脑正在疯狂尝试几十种参数组合 (耗时约几分钟，请耐心等待风扇狂转)...")
random_search.fit(X, y)

print("\n🎉 [4/4] 调优结束！最强版本答案揭晓：")
print("=========================================")
print(f"🏆 最佳参数配方: {random_search.best_params_}")
print(f"📈 最佳 F1-Score: {random_search.best_score_:.4f}")
print("=========================================")
print("你可以把这套参数直接填回你之前的全量代码里了！")