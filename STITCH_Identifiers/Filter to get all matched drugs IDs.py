import pandas as pd
from fuzzywuzzy import fuzz

# Load the TSV file
tsv_file = 'All_filtered_chemicals_file.tsv'
df_tsv = pd.read_csv(tsv_file, sep='\t')

# Load the CSV file
csv_file = 'Drug InChi Keys/All_drug_Inchi_and_smiles.csv'
df_csv = pd.read_csv(csv_file)

# Remove rows where the 'SMILES' column contains 'No SMILES found'
df_csv = df_csv[df_csv['SMILES'] != 'No SMILES found']

# Perform an exact match merge between the 'SMILES_string' column in the TSV and 'SMILES' column in the CSV
merged_df = pd.merge(df_tsv, df_csv, left_on='SMILES_string', right_on='SMILES', how='inner')

# Calculate similarity scores between the 'name' and 'Drug' columns
merged_df['similarity'] = merged_df.apply(lambda row: fuzz.ratio(str(row['name']), str(row['Drug'])), axis=1)

# Select the row with the highest similarity score for each InChI Key
filtered_df = merged_df.loc[merged_df.groupby('InChI Key')['similarity'].idxmax()]

# Save the final filtered DataFrame to a new CSV file
output_file = 'STITCH_Identifiers/All_Matched_Chemicals.csv'
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data with highest similarity score for each InChI Key saved to {output_file}")