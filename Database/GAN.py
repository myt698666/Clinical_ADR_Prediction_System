import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# Load your data
data = pd.read_csv('Database/merged_data.csv')

# Remove the first column (drug name) if it is non-numeric
data = data.iloc[:, 1:]  # Skip the first column

# Separate Inputs (X) and Outputs (y)
X = data.iloc[:, :-21].values  # Assuming the last 21 columns are the outputs
y = data.iloc[:, -21:].values  # Last 21 columns as the targets (outputs)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Normalize the input data
scaler = MinMaxScaler(feature_range=(0, 1))  # Scale data to range [-1, 1]
X_scaled = scaler.fit_transform(X)  # Normalize the input data

# Convert the data into tensors
X_tensor = torch.Tensor(X_scaled)
y_tensor = torch.Tensor(y)  # Convert the target data into a tensor

# Create a DataLoader for efficient batching
batch_size = 64  # Choose an appropriate batch size
dataset = TensorDataset(X_tensor, y_tensor)  # Include both inputs and targets
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,)

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, input_dim, z_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim + z_dim, 128),  # Combine inputs and noise
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),  # Output matches input dimensions
            nn.Tanh()  # Output values scaled to [-1, 1]
        )
    
    def forward(self, x, z):
        # Concatenate input data (x) with random noise (z)
        combined_input = torch.cat([x, z], dim=1)
        return self.fc(combined_input)

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability (real or fake)
        )
    
    def forward(self, x):
        return self.fc(x)

# Loss function and optimizers
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
lr = 0.0002  # Learning rate

# Training parameters
num_epochs = 1000  # Number of training epochs
z_dim = 100  # Latent space dimension (noise vector)
data_dim = X_scaled.shape[1]  # Dimension of your input data (number of features)

# Loop over each target column (output variable)
for output_index in range(y.shape[1]):
    print(f"Training GAN for Output {output_index + 1}")
    
    # Initialize the generator and discriminator
    generator = Generator(data_dim, z_dim, data_dim)  # Output matches input dimensions
    discriminator = Discriminator(data_dim)  # Set input_dim to the number of features

    # Optimizers for both models
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Training the GAN
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for real_data, _ in data_loader:  # Extract only the input features
            # Ensure real_data has the correct shape
            real_labels = torch.ones(real_data.size(0), 1)  # Batch size is dynamic
            
            # Generate fake data using the generator
            z = torch.randn(real_data.size(0), z_dim)  # Random noise vector
            fake_data = generator(real_data, z)  # Output shape: (batch_size, data_dim)
            fake_labels = torch.zeros(real_data.size(0), 1)  # Fake data labeled as 0
            
            # Train the Discriminator
            optimizer_D.zero_grad()
            
            # Real data loss
            real_loss = criterion(discriminator(real_data), real_labels)
            
            # Fake data loss
            fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)  # Detach to not update the generator
            
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train the Generator
            optimizer_G.zero_grad()
            
            # We want the generator to fool the discriminator, so use real labels for fake data
            g_loss = criterion(discriminator(fake_data), real_labels)
            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
    
    # Generate synthetic data after training
    z = torch.randn(1000, z_dim)  # Generate 1000 samples
    synthetic_data = generator(X_tensor[:1000], z)

    # Convert the synthetic data back to a readable format
    synthetic_data = synthetic_data.detach().numpy()

    # Inverse transform to bring the data back to the original scale
    synthetic_data_original_scale = scaler.inverse_transform(synthetic_data)

    # Option 1: Save all features
    columns = [f'Feature_{i + 1}' for i in range(data_dim)]  # Generate column names for all features
    synthetic_df = pd.DataFrame(synthetic_data_original_scale, columns=columns)

    # Option 2: Save only the specific output
    # synthetic_output = synthetic_data_original_scale[:, output_index].reshape(-1, 1)  # Shape (1000, 1)
    # synthetic_df = pd.DataFrame(synthetic_output, columns=[f'Output_{output_index + 1}'])

    # Save the DataFrame to a CSV file
    synthetic_df.to_csv(f'synthetic_data_output_{output_index + 1}.csv', index=False)

print("Synthetic data generation completed for all outputs!")
