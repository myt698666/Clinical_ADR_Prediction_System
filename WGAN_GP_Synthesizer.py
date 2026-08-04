import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Hyperparameters and Device Configuration
# ==========================================
BATCH_SIZE = 64
NOISE_DIM = 100
FEATURE_DIM = 500
LAMBDA_GP = 10
CRITIC_ITER = 5
EPOCHS = 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"System initialization. Target device: {DEVICE}")


# ==========================================
# 2. WGAN-GP Architecture Definition
# ==========================================
class Generator(nn.Module):
    def __init__(self, feature_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, feature_dim)
        )

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, feature_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand((real_samples.size(0), 1)).to(DEVICE)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones((real_samples.size(0), 1)).to(DEVICE)

    gradients = grad(
        outputs=d_interpolates, inputs=interpolates, grad_outputs=fake,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA_GP


# ==========================================
# 3. Data Ingestion
# ==========================================
print("Loading multi-modal clinical feature matrix.")
try:
    slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
    vip_features = slim_df.columns.tolist()
    inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv", index_col='Matched Drug')
    outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")

    target_soc = 'Ear'
    merged_data = inputs.merge(outputs[['Drug', target_soc]], how='left', left_index=True, right_on='Drug').dropna()
    minority_data = merged_data[merged_data[target_soc] == 1][vip_features].values

    feature_dim = minority_data.shape[1]
    batch_size = min(BATCH_SIZE, len(minority_data) // 2)

    dataloader = torch.utils.data.DataLoader(
        torch.tensor(minority_data, dtype=torch.float32).to(DEVICE),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
except FileNotFoundError:
    print("Error: Input data files not found.")
    dataloader = []

# ==========================================
# 4. Training Routine
# ==========================================
if dataloader:
    gen = Generator(feature_dim).to(DEVICE)
    critic = Critic(feature_dim).to(DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))

    print("Initiating WGAN-GP training routine.")
    for epoch in range(EPOCHS):
        for real in dataloader:
            # Training Critic
            for _ in range(CRITIC_ITER):
                noise = torch.randn(real.size(0), NOISE_DIM).to(DEVICE)
                fake = gen(noise)
                gp = compute_gradient_penalty(critic, real, fake)
                loss_critic = -(torch.mean(critic(real)) - torch.mean(critic(fake))) + gp

                opt_critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

            # Training Generator
            noise = torch.randn(real.size(0), NOISE_DIM).to(DEVICE)
            loss_gen = -torch.mean(critic(gen(noise)))

            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] | Critic Loss: {loss_critic.item():.4f} | Gen Loss: {loss_gen.item():.4f}")

    # Generation of synthetic features
    gen.eval()
    with torch.no_grad():
        synthetic_data = gen(torch.randn(500, NOISE_DIM).to(DEVICE)).cpu().numpy()
        syn_df = pd.DataFrame(synthetic_data, columns=vip_features)
        syn_df['Label'] = 1
        syn_df.to_csv(f"WGAN_Synthetic_{target_soc}_Features.csv", index=False)
        print(f"Synthetic samples exported to 'WGAN_Synthetic_{target_soc}_Features.csv'.")