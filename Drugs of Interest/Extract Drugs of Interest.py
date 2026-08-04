# Import necessary libraries
import pandas as pd

# Load input and output files
input_file_path = "STITCH_Identifiers/drug_protein_interaction_matrix.csv"
output_file_path = "ADR_Summary/SOC_significance_matrix.csv"

inputs = pd.read_csv(input_file_path, index_col=0)
outputs = pd.read_csv(output_file_path)

# Merge and clean data
merged_data = inputs.merge(outputs, how='left', left_on='Matched Drug', right_on='Drug')
merged_data = merged_data.dropna()

# Display the first 5 rows of the dataset
print(merged_data.head())

# Load the CSV file from the 'Feature Importance' folder
feature_file = "Feature Importance/feature_importance_Psych.csv"
df = pd.read_csv(feature_file)

# Extract the top 10 rows of the dataset
top_10_rows = df.head(20)

# Display the top 10 rows
print("Top 10 rows of the dataset:")
print(top_10_rows)

# Extract the 'Feature' column from the top 10 rows
top_10_features = top_10_rows['Feature'].tolist()

# Create 10 new datasets based on the top 10 features
for feature in top_10_features:
    if feature in merged_data.columns:
        # Extract the 'Drug', 'Psych', and feature columns
        new_dataset = merged_data[['Drug', 'Psych', feature]]
        
        # Save the new dataset to a CSV file (optional)
        new_dataset.to_csv(f"{feature}_dataset.csv", index=False)
        
        # Display the first 5 rows of the new dataset
        print(f"Dataset for feature: {feature}")
        print(new_dataset.head())
        
        # Analysis
        # 1. Count all non-zero values in both columns ('Psych' and feature)
        non_zero_psych = (new_dataset['Psych'] != 0).sum()
        non_zero_feature = (new_dataset[feature] != 0).sum()
        
        # 2. Count non-zero values that are in both columns for the same row
        non_zero_both = ((new_dataset['Psych'] != 0) & (new_dataset[feature] != 0)).sum()
        
        # 3. Create a DataFrame for non-zero values in the feature column along with drug names
        non_zero_feature_df = new_dataset.loc[new_dataset[feature] != 0, ['Drug', feature]]
        #non_zero_feature_df.to_csv(f"Drugs of Interest/Target associations/Interactions/{feature}_interactions.csv", index=False)  # Save to CSV
        
        # 4. Create a DataFrame for non-zero values in both columns for the same row along with drug names
        non_zero_both_df = new_dataset.loc[
            (new_dataset['Psych'] != 0) & (new_dataset[feature] != 0), ['Drug', 'Psych', feature]
        ]
        #non_zero_both_df.to_csv(f"Drugs of Interest/Target associations/Matching Signals/{feature}_matching signals.csv", index=False)  # Save to CSV
        
        # Display the results
        print(f"Analysis for feature: {feature}")
        print(f"Non-zero values in 'Psych': {non_zero_psych}")
        print(f"Non-zero values in '{feature}': {non_zero_feature}")
        print(f"Non-zero values in both columns (same row): {non_zero_both}")
        print(f"Non-zero values in '{feature}' with drug names saved to {feature}_non_zero_feature.csv")
        print(f"Non-zero values in both columns (same row) with drug names saved to {feature}_non_zero_both.csv")
    else:
        print(f"Feature '{feature}' not found in merged_data columns.")