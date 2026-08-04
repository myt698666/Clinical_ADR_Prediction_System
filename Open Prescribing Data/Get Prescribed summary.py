import pandas as pd

# Load the dataset
file_path = 'Open Prescribing Data/drug_data.csv'  # Update the file path if needed
df = pd.read_csv(file_path)

# Group by 'drug_name' and aggregate the 'items' column
grouped_data = df.groupby('drug_name')['items'].sum().reset_index()

# Save the result to 'Open Prescribing Data/prescribed summary.csv'
output_file_path = 'Open Prescribing Data/prescribed summary.csv'
grouped_data.to_csv(output_file_path, index=False)

# Show the first few rows of the grouped data
print(grouped_data.head())
