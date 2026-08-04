import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')

print("🚀 [1/4] 正在加载 500 维黄金燃料...")
slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
vip_features = slim_df.columns.tolist()

inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv", index_col='Matched Drug')
outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug').dropna()

X = merged_data[vip_features].values
output_columns = outputs.columns.drop('Drug')

print(f"🎯 [2/4] 燃料加载完毕！即将对全部 {len(output_columns)} 个副作用大类进行 5 折交叉验证扫荡！\n")

# 准备一个空列表，用来装所有副作用的最终成绩
final_results = []

# 开始循环扫荡所有的副作用
for target in output_columns:
    y = merged_data[target].values

    # 有些罕见的副作用可能一个正样本都没有，遇到这种情况直接跳过
    if sum(y == 1) < 5:
        print(f"⚠️ 跳过 '{target}'：正样本太少，无法进行 5 折交叉验证。")
        continue

    print(f"⏳ 正在攻克目标 '{target}' ...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_list, auprc_list, recall_list = [], [], []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        pos_count = sum(y_train == 1)
        neg_count = sum(y_train == 0)
        scale_weight = neg_count / pos_count if pos_count > 0 else 1

        # 搭载我们的冠军参数引擎
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.9, scale_pos_weight=scale_weight,
            random_state=42, n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

        f1_list.append(f1_score(y_test, y_pred, zero_division=0))
        auprc_list.append(average_precision_score(y_test, y_pred_proba))
        recall_list.append(recall_score(y_test, y_pred, zero_division=0))

    # 把这个副作用的 5 次平均分记录下来
    final_results.append({
        'SOC_Target': target,
        'Recall_Mean': np.mean(recall_list),
        'Recall_Std': np.std(recall_list),
        'F1_Score_Mean': np.mean(f1_list),
        'F1_Score_Std': np.std(f1_list),
        'AUPRC_Mean': np.mean(auprc_list),
        'AUPRC_Std': np.std(auprc_list)
    })

print("\n📊 [3/4] 所有目标扫荡完毕！正在生成终极成绩单...")

# 将结果转换成漂亮的表格
results_df = pd.DataFrame(final_results)
# 按照 F1 分数从高到低排序，看看哪个副作用我们预测得最准
results_df = results_df.sort_values(by='F1_Score_Mean', ascending=False)

# 保存最终结果
output_csv_path = "ADR_Summary/Final_27_SOC_CV_Results.csv"
results_df.to_csv(output_csv_path, index=False)

print(f"💾 [4/4] 大功告成！全量成绩单已完美保存至: {output_csv_path}")
print("\n🏆 排名前 3 的最好预测目标是：")
print(results_df[['SOC_Target', 'Recall_Mean', 'F1_Score_Mean']].head(3).to_string(index=False))