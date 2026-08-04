import pandas as pd

# Load the CSV file
file_path = "ADR_Summary/All_ADR_Data.csv"
df = pd.read_csv(file_path)

# Extract unique values from PT column
unique_pt_values = df['PT'].unique()

# Create a mapping DataFrame
mapping_df = df[['PT', 'HLT', 'HLGT', 'SOC_ABBREV']].drop_duplicates()

# Save the mapping to a new CSV file
output_file = "ADR_Summary/ADR_MedDRA_Key_Mapping.csv"
mapping_df.to_csv(output_file, index=False)

print(f"Key mapping file saved to {output_file}.")
