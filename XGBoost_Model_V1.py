import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
import warnings

# 忽略烦人的版本警告
warnings.filterwarnings('ignore')

print("🚀 [1/3] 正在读取黄金 VIP 名单并加载燃料...")
# 1. 读取瘦身版，纯粹为了获取那 500 个黄金特征的名字 (VIP 名单)
slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
vip_features = slim_df.columns.tolist()

# 2. 读取原始的超级矩阵（带有完整的药物名字，绝不报错）
input_file_path = "STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col='Matched Drug')
outputs = pd.read_csv(output_file_path)

# 3. 完美拼接对齐特征和标签
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')
merged_data = merged_data.dropna()

# 4. 💡 核心降维打击：用 VIP 名单直接过滤，瞬间将 2548 维缩减为最强 500 维！
X = merged_data[vip_features]
output_columns = outputs.columns.drop('Drug')

print(f"🧬 特征维度加载完毕: 药物数量 {X.shape[0]} 个, 特征高达 {X.shape[1]} 维")
print("🔥 [2/3] 启动 XGBoost 涡轮引擎 (已开启自动类别平衡与 CPU 全核加速)...\n")

# 循环训练所有 27 个 SOC 副作用大类
for target in output_columns:
    y = merged_data[target]

    # 切分训练集和测试集 (80% 训练, 20% 考试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 💡 核心降维打击：动态计算类别权重，直接秒杀“数据不平衡”问题！
    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    # 如果正样本极少，惩罚权重就会非常高
    scale_weight = neg_count / pos_count if pos_count > 0 else 1

    # 初始化 XGBoost 模型
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,             # 建立 100 棵树
        max_depth=6,                  # 每棵树的最大深度
        learning_rate=0.1,            # 学习率
        scale_pos_weight=scale_weight,# 自动火力倾斜，拯救低 F1 分数
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1                     # -1 表示榨干 CPU 的所有核心
    )

    # 训练模型
    xgb_model.fit(X_train, y_train)

    # 预测标签和概率
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    # 评估终极成绩
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auprc = average_precision_score(y_test, y_pred_proba)

    # 打印每个副作用分类的成绩单
    print(f"🎯 Target '{target}': Accuracy={acc:.2f} | Precision={prec:.2f} | Recall={rec:.2f} | F1-Score={f1:.2f} | AUPRC={auprc:.2f}")

print("\n🎉 [3/3] 降维打击完成！所有模型训练结束！")