import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

# --- Bayesian Uncertainty Quantification via MC-Dropout ---
def predict_with_uncertainty(model, X_scaled, n_iter=20):
    """
    Performs Monte Carlo Dropout inference to quantify epistemic uncertainty.
    """
    model.train()  # Enable dropout layers for stochastic inference
    preds = []
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    for _ in range(n_iter):
        with torch.no_grad():
            out = model(X_t)
            preds.append(out.cpu().numpy())
    preds = np.array(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std


def get_confidence_label(std, threshold=0.1):
    """
    Categorizes predictions into confidence levels based on uncertainty threshold.
    """
    return ["High Confidence" if s < threshold else "Low Confidence" for s in std]


# --- Visualization Module ---
def plot_confidence_distribution(all_confidences):
    """
    Generates a bar chart illustrating the distribution of prediction confidence.
    """
    high_count = all_confidences.count('High Confidence')
    low_count = all_confidences.count('Low Confidence')
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['High Confidence', 'Low Confidence'], [high_count, low_count], color=['#2ecc71', '#e74c3c'])
    plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Samples')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, yval, ha='center', va='bottom')
    plt.savefig('Confidence_Distribution.png', dpi=300)
    print("\nVisualization chart saved as 'Confidence_Distribution.png'.")
    plt.show()


# --- Main Execution Logic ---
warnings.filterwarnings('ignore')
print("Initializing Master Ensemble Engine with Uncertainty Quantification.")

# Load data
try:
    slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
    vip_features = slim_df.columns.tolist()
    inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv", index_col='Matched Drug')
    outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")
    merged_data = inputs.merge(outputs[['Drug', 'Neopl']], how='left', left_on=inputs.index, right_on='Drug').dropna()
    X = merged_data[vip_features].values
    y = merged_data['Neopl'].values.astype(int)
except Exception as e:
    print(f"Error: Data loading failed. {e}")
    exit()


class ClinicalMLP(nn.Module):
    def __init__(self, input_dim):
        super(ClinicalMLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
                                 nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x): return self.net(x)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res_ens = {'f1': [], 'rec': []}
all_confidences = []
device = torch.device("cpu")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = ClinicalMLP(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Training phase
    model.train()
    X_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    for _ in range(50):
        optimizer.zero_grad()
        criterion(model(X_t), y_t).backward()
        optimizer.step()

    # Analysis phase
    mean_preds, std = predict_with_uncertainty(model, X_test_scaled)
    confidences = get_confidence_label(std)
    all_confidences.extend(confidences)

    pred_ens = (mean_preds >= 0.5).astype(int)
    res_ens['f1'].append(f1_score(y_test, pred_ens, zero_division=0))
    print(f"Fold {fold} analysis complete: {confidences.count('High Confidence')} high confidence predictions.")

plot_confidence_distribution(all_confidences)
print("\nFinal Ensemble Report: Mean F1-Score: {:.4f}".format(np.mean(res_ens['f1'])))