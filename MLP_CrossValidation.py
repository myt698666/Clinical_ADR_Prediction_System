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

# ==========================================
# 1. System Initialization
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computational engine initialized: {device}")

# Data Loading
print("Loading 500-dimensional VIP feature matrix.")
slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
vip_features = slim_df.columns.tolist()

inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv", index_col='Matched Drug')
outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug').dropna()

# Feature/Target segregation
X = merged_data[vip_features].values.astype(np.float32)
target = 'Blood'
y = merged_data[target].values.astype(np.float32)

# ==========================================
# 2. Network Architecture: MLP for ADR Prediction
# ==========================================
class AdverseReactionMLP(nn.Module):
    """
    Multilayer Perceptron architecture for high-dimensional clinical feature classification.
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
# 3. Model Training and Evaluation (5-Fold Cross-Validation)
# ==========================================
print(f"Initiating 5-Fold Stratified Cross-Validation for target: '{target}'.")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores, auprc_scores, recall_scores = [], [], []

fold = 1
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Dynamic class weighting for imbalance mitigation
    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    pos_weight = torch.tensor([neg_count / pos_count]).to(device) if pos_count > 0 else torch.tensor([1.0]).to(device)

    # Data transformation for PyTorch ingestion
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Model configuration
    model = AdverseReactionMLP(input_dim=X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

    # Training Routine
    model.train()
    for epoch in range(20):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs_pred = model(batch_X)
            loss = criterion(outputs_pred, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation Routine
    model.eval()
    with torch.no_grad():
        test_X_tensor = torch.tensor(X_test).to(device)
        raw_outputs = model(test_X_tensor)
        y_pred_proba = torch.sigmoid(raw_outputs).cpu().numpy().flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

    f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
    auprc_scores.append(average_precision_score(y_test, y_pred_proba))
    recall_scores.append(recall_score(y_test, y_pred, zero_division=0))

    print(f"Fold {fold} | Recall: {recall_scores[-1]:.2f} | F1-Score: {f1_scores[-1]:.2f} | AUPRC: {auprc_scores[-1]:.2f}")
    fold += 1

# ==========================================
# 4. Final Performance Report
# ==========================================
print("\nBenchmark Evaluation Results:")
print("--------------------------------------------------")
print(f"Mean Recall:    {np.mean(recall_scores):.4f} (±{np.std(recall_scores):.4f})")
print(f"Mean F1-Score:  {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
print(f"Mean AUPRC:     {np.mean(auprc_scores):.4f} (±{np.std(auprc_scores):.4f})")
print("--------------------------------------------------")