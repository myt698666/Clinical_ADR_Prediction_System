import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import warnings

warnings.filterwarnings('ignore')

print("🧬 ========================================================")
print("🚀 [Track 2] WGAN-GP Minority Feature Synthesizer Init...")
print("🧬 ========================================================\n")

# ==========================================
# 1. 超参数与设备配置
# ==========================================
BATCH_SIZE = 64
NOISE_DIM = 100
FEATURE_DIM = 500  # 我们的黄金多模态矩阵维度
LAMBDA_GP = 10  # 梯度惩罚系数
CRITIC_ITER = 5  # 每训练1次生成器，训练5次判别器
EPOCHS = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Using Device: {device}")


# ==========================================
# 2. 构建 WGAN-GP 架构 (适用于结构化表格数据)
# ==========================================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, FEATURE_DIM)
            # 注意：表格数据不需要 Tanh，因为我们的特征可能不在 [-1, 1] 之间
        )

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):  # WGAN 中称为 Critic 而不是 Discriminator
    def __init__(self):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # 不加 Sigmoid
        )

    def forward(self, x):
        return self.net(x)


# 梯度惩罚计算函数 (Gradient Penalty)
def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones((real_samples.size(0), 1)).to(device)

    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA_GP
    return gradient_penalty


# ==========================================
# 3. 数据加载 (提取极度罕见靶点的阳性样本)
# ==========================================
print("⏳ Loading multi-modal features...")
try:
    # --- 修复核心：先读取我们提纯好的 500 维黄金特征列表 ---
    slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
    vip_features = slim_df.columns.tolist()  # 获取这 500 个特征的名字

    # 读取完整的矩阵和标签
    inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv", index_col='Matched Drug')
    outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")

    # 提取一个非常不平衡的标签，比如 Ear (耳部毒性) 或 Repro (生殖系统)
    target_soc = 'Ear'
    merged_data = inputs.merge(outputs[['Drug', target_soc]], how='left', left_index=True, right_on='Drug').dropna()

    # 【核心修复】：只保留 vip_features 这 500 列，确保维度匹配
    minority_data_df = merged_data[merged_data[target_soc] == 1][vip_features]
    minority_data = minority_data_df.values

    # 获取实际的特征维度（现在应该是 500）
    ACTUAL_FEATURE_DIM = minority_data.shape[1]
    # 更新全局 FEATURE_DIM，以防万一不是严格的 500
    FEATURE_DIM = ACTUAL_FEATURE_DIM

    # 如果阳性样本少于批量大小，动态调整 BATCH_SIZE
    if len(minority_data) < BATCH_SIZE:
        BATCH_SIZE = len(minority_data) // 2

    real_data_tensor = torch.tensor(minority_data, dtype=torch.float32).to(device)
    # 创建简单的数据加载器
    dataloader = torch.utils.data.DataLoader(real_data_tensor, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print(f"🎯 Target SOC: {target_soc}")
    print(f"📊 Minority Class Samples loaded: {len(minority_data)}")
    print(f"📏 Feature Dimension aligned to: {FEATURE_DIM}")
except FileNotFoundError:
    print("❌ Error: Missing data files. (Skipping training loop for demonstration)")
    dataloader = []

# ==========================================
# 4. 初始化模型与优化器
# ==========================================
# (由于我们在步骤3可能动态更新了 FEATURE_DIM，必须在这里重新实例化模型)
gen = Generator().to(device)
critic = Critic().to(device)

opt_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))

# ==========================================
# 5. 训练循环 (WGAN-GP)
# ==========================================
if dataloader:
    print("⚡ Starting WGAN-GP Training...")
    for epoch in range(EPOCHS):
        for batch_idx, real in enumerate(dataloader):
            cur_batch_size = real.shape[0]

            # ---------------------
            # 训练 Critic (判别器)
            # ---------------------
            for _ in range(CRITIC_ITER):
                noise = torch.randn(cur_batch_size, NOISE_DIM).to(device)
                fake = gen(noise)

                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = compute_gradient_penalty(critic, real, fake)

                # WGAN Loss
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + gp

                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # ---------------------
            # 训练 Generator (生成器)
            # ---------------------
            noise = torch.randn(cur_batch_size, NOISE_DIM).to(device)
            fake = gen(noise)
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        if (epoch + 1) % 50 == 0:
            print(
                f"   Epoch [{epoch + 1}/{EPOCHS}] | Critic Loss: {loss_critic.item():.4f} | Gen Loss: {loss_gen.item():.4f}")

    # ==========================================
    # 6. 生成并保存合成的虚拟特征数据
    # ==========================================
    print("\n✨ Training Complete! Generating synthetic minority features...")
    gen.eval()
    with torch.no_grad():
        num_synthetic_samples = 500  # 我们希望合成 500 个假的阳性样本来平衡数据
        z = torch.randn(num_synthetic_samples, NOISE_DIM).to(device)
        synthetic_features = gen(z).cpu().numpy()

    # 修复：这里的列名必须使用我们提纯后的 500 维黄金特征 (vip_features)
    syn_df = pd.DataFrame(synthetic_features, columns=vip_features)
    syn_df['Label'] = 1  # 这些全是合成的阳性样本
    syn_df.to_csv(f"WGAN_Synthetic_{target_soc}_Features.csv", index=False)
    print(f"💾 Saved {num_synthetic_samples} synthetic samples to 'WGAN_Synthetic_{target_soc}_Features.csv'.")
    print("🚀 Next step: Blend this synthetic data with real data and retrain XGBoost/MLP!")