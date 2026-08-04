import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error

# Load the data
file_path = 'Drug analysis.xlsx'  # Path to your file
sheet_name = 'Final With Norm'

# Read the data from the Excel file
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Selecting input features and target
input_columns = [
    'Molecular Weight', 'AlogP', 'Aromatic Rings', 'Cx LogD',
    'Cx LogP', 'Cx Most Bpka', 'HBA', 'HBD', 'HBA Lipinski',
    'HBD Lipinski', 'Heavy Atoms', 'Rotatable Bonds', 'TPSA', 'Confidence'
]
output_column = 'ADR Norm (Normalized)'

X = df[input_columns]
y = df[output_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer with a single neuron since it's regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on test data
mse_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Mean Squared Error on Test Set: {mse_loss[0]}")

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Optional: Save the model if you wish
model.save('drug_adr_nn_model.h5')

# Optionally, plot the training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(history.history['mse'], label='Training MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Model Training History')
plt.legend()
plt.grid(True)
plt.show()
