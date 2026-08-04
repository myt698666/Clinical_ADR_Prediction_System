import pandas as pd

# Paths to the files
drugs_file = "InBirg DDPD Values/ddpd-drugs.csv"
keys_file = "Drugs and keys/Drug Master Keys.csv"
activity_file = "Drug Activity/chEMBL_activity_values.csv"
output_file = "Drug Activity/filtered_activity_values.csv"

# Load the datasets
drugs_df = pd.read_csv(drugs_file)
keys_df = pd.read_csv(keys_file)
activity_df = pd.read_csv(activity_file)

# Group by 'Drug-Name' and calculate the average of 'C-Max (nM)'
grouped_drugs_df = drugs_df.groupby('Drug-Name', as_index=False)['C-Max (nM)'].mean()

# Perform an inner join with the drug master keys on 'Drug-Name' and 'Drug'
merged_df = grouped_drugs_df.merge(keys_df, left_on='Drug-Name', right_on='Drug')

# Perform an inner join with the activity file on 'chembl_id' and 'Molecule ChEMBL ID'
final_df = merged_df.merge(activity_df, left_on='chembl_id', right_on='Molecule ChEMBL ID')

# Remove unwanted columns
columns_to_remove = [
    'Drug', 'PubChem CID', 'InChI Key', 'SMILES', 
    'STITCH ID', 'BNF Code', 'chembl_id', 
    'atc_codes', 'Molecule Name'
]
filtered_df = final_df.drop(columns=columns_to_remove, errors='ignore')

# Add a column for Standard Value divided by C-Max (nM)
filtered_df['Standard/C-Max'] = filtered_df['Standard Value'] / filtered_df['C-Max (nM)']

# Handle NaN values by filling them with a default value (e.g., 0 or another placeholder)
filtered_df['Standard/C-Max'] = filtered_df['Standard/C-Max'].fillna(0)

# Add a classification column based on the computed value
filtered_df['Value Category'] = pd.cut(
    filtered_df['Standard/C-Max'],
    bins=[-float('inf'), 1, 5, float('inf')],
    labels=[2, 1, 0]
).astype(int)

# Save the result to a CSV file
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data saved to '{output_file}'")
