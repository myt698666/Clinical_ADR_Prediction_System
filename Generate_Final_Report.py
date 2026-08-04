import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 读取 XGBoost 和 MLP 的两份成绩单
# ==========================================
print(" Reading the scorecards for XGBoost and MLP...")
# 读取 XGBoost 成绩单 (根据你截图里的列名，包含 SOC_Target, F1_Score_Mean, Recall_Mean)
xgb_df = pd.read_csv("ADR_Summary/Final_27_SOC_CV_Results.csv")

# 读取 MLP 成绩单 (包含 SOC_Target, MLP_F1_Mean, MLP_Recall_Mean)
mlp_df = pd.read_csv("ADR_Summary/MLP_All_Targets_Results.csv")

# ==========================================
# 2. 合并表格，生成终极对比大表
# ==========================================
# 按靶点(SOC_Target)进行合并
merged_df = pd.merge(xgb_df, mlp_df, on='SOC_Target', how='inner')

# 计算 F1 分数的提升/下降幅度
merged_df['F1_Difference (MLP - XGB)'] = merged_df['MLP_F1_Mean'] - merged_df['F1_Score_Mean']

# 按照 XGBoost 的 F1 分数降序排列，让图表更好看
merged_df = merged_df.sort_values(by='F1_Score_Mean', ascending=False)

# 保存最终合并的表格
output_csv = "ADR_Summary/Ultimate_Model_Comparison.csv"
merged_df.to_csv(output_csv, index=False)
print(f" {output_csv}")

print("preparing")

targets = merged_df['SOC_Target'].tolist()
xgb_f1 = merged_df['F1_Score_Mean'].tolist()
mlp_f1 = merged_df['MLP_F1_Mean'].tolist()

x = np.arange(len(targets))  # 靶点标签的横坐标位置
width = 0.35  # 柱子的宽度

fig, ax = plt.subplots(figsize=(14, 7)) # 设置画布大小


rects1 = ax.bar(x - width/2, xgb_f1, width, label='XGBoost (Optimized Tree)', color='#1f77b4', edgecolor='black')
rects2 = ax.bar(x + width/2, mlp_f1, width, label='MLP (Deep Learning)', color='#ff7f0e', edgecolor='black')

ax.set_ylabel('F1-Score (5-Fold CV Mean)', fontsize=12, fontweight='bold')
ax.set_xlabel('MedDRA System Organ Class (SOC)', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison: XGBoost vs. Deep Learning (MLP) across 22 ADR Targets', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=11)
ax.legend(fontsize=12)


ax.grid(axis='y', linestyle='--', alpha=0.7)

for i, target in enumerate(targets):
    if target in ['Neopl', 'Repro']:
        ax.get_xticklabels()[i].set_color("red")
        ax.get_xticklabels()[i].set_fontweight("bold")

fig.tight_layout()

# 保存图表
output_img = "ADR_Summary/Model_Comparison_BarChart.png"
plt.savefig(output_img, dpi=300) # 300 dpi 是符合学术期刊要求的高清分辨率
print(f" High-definition comparison bar chart has been saved: {output_img}")
plt.show()
