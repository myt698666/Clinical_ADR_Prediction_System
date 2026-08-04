import pandas as pd

# Load the CSV files into pandas DataFrames
df_drugs = pd.read_csv('Drug InChi Keys/Drug InChi Keys/All_drug_Inchi_and_smiles_with_chembl.csv.csv')
df_stitch = pd.read_csv('STITCH_Identifiers/All_Matched_Chemicals.csv')
df_bnf = pd.read_csv('Open Prescribing/bnf/grouped_bnf_codes.csv')

# Merge the first two dataframes on SMILES (left join)
merged_df = pd.merge(df_drugs, df_stitch[['SMILES_string', 'chemical']], left_on='SMILES', right_on='SMILES_string', how='left')

# Merge with the third dataframe on 'Drug' (left join)
merged_df = pd.merge(merged_df, df_bnf[['Matched Drug', 'BNF Code']], left_on='Drug', right_on='Matched Drug', how='left')

# Rename the 'chemical' column to 'STITCH ID'
merged_df = merged_df.rename(columns={'chemical': 'STITCH ID'})

# Select the columns you want to keep in the final dataframe, including 'PubChem CID', 'chembl_id', and 'atc_codes'
final_df = merged_df[['Drug', 'PubChem CID', 'InChI Key', 'SMILES', 'STITCH ID', 'BNF Code', "ChEMBL ID","ATC Code"]]

# Drop duplicates based on the 'Drug' column
final_df = final_df.drop_duplicates(subset=['Drug'])

# Save the final dataframe to a CSV file
final_df.to_csv('Drugs and keys/Drug Master Keys.csv', index=False)

print("File saved as 'Drugs and keys/Drug Master Keys.csv'")
