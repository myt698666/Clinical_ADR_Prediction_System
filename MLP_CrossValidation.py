import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')

# 检查电脑有没有显卡（GPU），有的话就开启加速，没有就用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 [1/4] 当前计算引擎锁定为: {device}")

print("⏳ [2/4] 加载 500 维黄金 VIP 燃料...")
slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
vip_features = slim_df.columns.tolist()

inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv", index_col='Matched Drug')
outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug').dropna()

X = merged_data[vip_features].values.astype(np.float32)
target = 'Blood'  # 继续拿血液系统当打擂台的试验田
y = merged_data[target].values.astype(np.float32)


# =======================================================
# 🧠 3. 建立多层感知机 (MLP) 神经网络架构
# =======================================================
class AdverseReactionMLP(nn.Module):
    def __init__(self, input_dim):
        super(AdverseReactionMLP, self).__init__()
        # 3层神经网络：500 维输入 -> 128 维隐藏 -> 32 维精简 -> 1 维输出分类
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),  # 批归一化，防止神经网络死掉
            nn.ReLU(),  # 激活函数，挖掘非线性关系
            nn.Dropout(0.3),  # 随机丢弃 30% 神经元，强力防止过拟合

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 1)  # 最终输出一个概率分数
        )

    def forward(self, x):
        return self.network(x)


print(f"🎯 [3/4] 神经网络构建完毕。开启针对 '{target}' 的深度学习【5 折交叉验证】...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores, auprc_scores, recall_scores = [], [], []

fold = 1
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 计算深度学习中的类别平衡权重
    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    # 类似 XGBoost 的 scale_pos_weight
    pos_weight = torch.tensor([neg_count / pos_count]).to(device) if pos_count > 0 else torch.tensor([1.0]).to(device)

    # 转换为 PyTorch 专用的张量数据集格式
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 初始化模型、损失函数与优化器
    model = AdverseReactionMLP(input_dim=X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 注入非平衡惩罚
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)  # 采用业界最强的 AdamW 优化器

    # 🎬 开始训练（跑 20 轮考试）
    model.train()
    for epoch in range(20):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # 📝 闭卷测试阶段
    model.eval()
    with torch.no_grad():
        test_X_tensor = torch.tensor(X_test).to(device)
        raw_outputs = model(test_X_tensor)
        # 用 Sigmoid 把输出压到 0~1 的概率区间
        y_pred_proba = torch.sigmoid(raw_outputs).cpu().numpy().flatten()
        # 概率 > 0.5 判定为引发副作用
        y_pred = (y_pred_proba > 0.5).astype(int)

    # 计算并记录本折的成绩
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auprc = average_precision_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred, zero_division=0)

    f1_scores.append(f1)
    auprc_scores.append(auprc)
    recall_scores.append(recall)

    print(f"   🔄 [Fold {fold}/5] 神经网络跑分: Recall={recall:.2f} | F1={f1:.2f} | AUPRC={auprc:.2f}")
    fold += 1

print("\n🎉 [4/4] 深度学习(MLP)大考结束！挤干水分后的真实战绩：")
print("==================================================")
print(f"🧠 神经网络 平均 Recall:    {np.mean(recall_scores):.4f} (±{np.std(recall_scores):.4f})")
print(f"🧠 神经网络 平均 F1-Score:  {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
print(f"🧠 神经网络 平均 AUPRC:     {np.mean(auprc_scores):.4f} (±{np.std(auprc_scores):.4f})")
print("==================================================")