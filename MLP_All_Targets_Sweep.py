import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score, recall_score
import warnings
import time
import os

warnings.filterwarnings('ignore')

# ==========================================
# 1. System Initialization
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computational engine initialized: {device}")

os.makedirs("ADR_Summary", exist_ok=True)

# ==========================================
# 2. Data Loading and Feature Preparation
# ==========================================
print("Loading 500-dimensional VIP feature matrix.")
slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
vip_features = slim_df.columns.tolist()

inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv", index_col='Matched Drug')
outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")

# ==========================================
# 3. Network Architecture: MLP for ADR Prediction
# ==========================================
class AdverseReactionMLP(nn.Module):
    """
    Multilayer Perceptron architecture for high-dimensional feature classification.
    """
    def __init__(self, input_dim):
        super(AdverseReactionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 4. Experimental Benchmarking
# ==========================================
targets = outputs.columns[1:]
results = []
print(f"Starting systematic benchmarking across {len(targets)} MedDRA targets.")

start_time = time.time()

for target in targets:
    target_data = inputs.merge(outputs[['Drug', target]], how='left', left_index=True, right_on='Drug').dropna()
    X = target_data[vip_features].values.astype(np.float32)
    y = target_data[target].values.astype(np.float32)

    # Filtering targets with insufficient positive samples for robust cross-validation
    if sum(y == 1) < 5:
        print(f"Skipping target '{target}': Insufficient positive samples.")
        continue

    print(f"Benchmarking target: '{target}'")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1, fold_auprc, fold_recall = [], [], []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Dynamic class weighting for imbalance mitigation
        pos_count = sum(y_train == 1)
        neg_count = sum(y_train == 0)
        pos_weight = torch.tensor([neg_count / pos_count]).to(device) if pos_count > 0 else torch.tensor([1.0]).to(device)

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        model = AdverseReactionMLP(input_dim=X.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

        # Training routine
        model.train()
        for epoch in range(20):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs_pred = model(batch_X)
                loss = criterion(outputs_pred, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluation routine
        model.eval()
        with torch.no_grad():
            test_X_tensor = torch.tensor(X_test).to(device)
            raw_outputs = model(test_X_tensor)
            y_pred_proba = torch.sigmoid(raw_outputs).cpu().numpy().flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

        fold_f1.append(f1_score(y_test, y_pred, zero_division=0))
        fold_auprc.append(average_precision_score(y_test, y_pred_proba))
        fold_recall.append(recall_score(y_test, y_pred, zero_division=0))

    results.append({
        'SOC_Target': target,
        'MLP_Recall_Mean': np.mean(fold_recall),
        'MLP_Recall_Std': np.std(fold_recall),
        'MLP_F1_Mean': np.mean(fold_f1),
        'MLP_F1_Std': np.std(fold_f1),
        'MLP_AUPRC_Mean': np.mean(fold_auprc),
        'MLP_AUPRC_Std': np.std(fold_auprc)
    })

# ==========================================
# 5. Result Export and Final Reporting
# ==========================================
end_time = time.time()
print(f"Benchmarking completed. Duration: {(end_time - start_time) / 60:.2f} minutes.")

results_df = pd.DataFrame(results).sort_values(by='MLP_F1_Mean', ascending=False)
output_path = "ADR_Summary/MLP_All_Targets_Results.csv"
results_df.to_csv(output_path, index=False)

print(f"Results exported to: {output_path}")
print("Top 3 targets by performance:")
print(results_df[['SOC_Target', 'MLP_Recall_Mean', 'MLP_F1_Mean']].head(3).to_string(index=False))