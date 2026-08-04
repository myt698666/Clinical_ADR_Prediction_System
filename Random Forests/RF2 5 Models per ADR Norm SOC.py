import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Step 1: Load the data
# Assuming the input data (X) is stored in a CSV file and the target (y) in another CSV file.
input_file_path = "STITCH_Identifiers/top_500_protein_drug_interaction_matrix.csv"
output_file_path = "ADR_Summary/Filtered_Normalized_matrix.csv"

# Load datasets
inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

# Step 2: Preprocess the data
# Merge the inputs and outputs (making sure they match on a common column, such as 'Drug')
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')

# Drop rows with missing output values
merged_data = merged_data.dropna()

# Separate features (X) and target columns (y)
X = merged_data.iloc[:, :-len(outputs.columns)]  # Features
output_columns = outputs.columns.drop('Drug')    # Targets (excluding 'Drug' column)

# Step 3: Split the data into training and testing sets
# We will train one model for each target column in the output.
results = []
for target in output_columns:
    y = merged_data[target]

    # Split into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Scale the data (optional but often beneficial)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Step 5: Initialize and train the Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # 100 trees
    rf_model.fit(X_train_scaled, y_train)

    # Step 6: Save the model (optional)
    model_filename = f"saved_models/SOC_Norm/random_forest_model_{target}.pkl"
    joblib.dump(rf_model, model_filename)

    # Step 7: Make predictions on the test set
    y_pred = rf_model.predict(X_test_scaled)

    # Step 8: Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Target': target,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Model': rf_model,
        'y_test': y_test,
        'y_pred': y_pred
    })

    print(f"Model trained for target '{target}' - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Step 9: (Optional) Analyze results for all targets
# For example, sort by R² to see which model performed best
sorted_results = sorted(results, key=lambda x: x['R2'], reverse=True)
for result in sorted_results:
    print(f"Target: {result['Target']} - R²: {result['R2']:.2f}")
