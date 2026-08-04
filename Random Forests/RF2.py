import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the input (top 500 protein matrix) and output (ADR significance matrix) files
input_file_path = "STITCH_Identifiers/top_500_protein_drug_interaction_matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

# Perform a left join on the 'Drug' column
merged_data = inputs.merge(outputs, how='left', left_index=True, right_on='Drug')

# Drop rows with missing output values (for all target columns)
merged_data = merged_data.dropna()

# Separate features (inputs) and target columns
X = merged_data.iloc[:, :-len(outputs.columns)]  # Inputs (all columns in `inputs`)
output_columns = outputs.columns.drop('Drug')    # Exclude the 'Drug' column
results = []

# Train a Random Forest model for each output column
for target in output_columns:
    y = merged_data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Store results for comparison
    results.append({
        'Target': target,
        'Accuracy': accuracy,
        'y_test': y_test.values,
        'y_pred': y_pred
    })
    
    print(f"Model trained for target '{target}' with accuracy: {accuracy:.2f}")

# Sort results by accuracy and select the top 5
top_5_results = sorted(results, key=lambda x: x['Accuracy'], reverse=True)[:10]

# Generate heatmaps for the top 5 targets
plt.figure(figsize=(20, 15))
for idx, result in enumerate(top_5_results):
    comparison_df = pd.DataFrame({
        'Actual': result['y_test'],
        'Predicted': result['y_pred']
    })

    plt.subplot(2, 5, idx + 1)  # 2 rows, 3 columns
    sns.heatmap(
        comparison_df.T, 
        cmap='coolwarm', 
        cbar=True, 
        xticklabels=False, 
        yticklabels=['Actual', 'Predicted']
    )
    plt.title(f"Heatmap for Target: {result['Target']}\nAccuracy: {result['Accuracy']:.2f}")

plt.tight_layout()
plt.show()
