import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys

# Load the dataset
file_path = "Drugs and keys/Drug Master Keys.csv"
df = pd.read_csv(file_path)
# Remove rows where the SMILES column has 'No SMILES found'
df = df[df["SMILES"] != "No SMILES found"]

# Ensure necessary columns exist
if 'SMILES' not in df.columns or 'Drug' not in df.columns:
    raise ValueError("The CSV file must contain 'Drug' and 'SMILES' columns.")

# Function to compute MACCS keys
def get_maccs_keys(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        maccs_fp = list(MACCSkeys.GenMACCSKeys(mol))  # Extract full 167 bits
        return maccs_fp[1:]  # Drop the first bit (not used)
    return [None] * 166  # Placeholder for invalid SMILES

# Compute MACCS keys for each row
df["MACCS_Keys"] = df["SMILES"].apply(get_maccs_keys)

# Convert MACCS key lists into separate columns
maccs_df = pd.DataFrame(df["MACCS_Keys"].to_list(), columns=[f"MACCS_{i}" for i in range(166)])

# Combine with original dataframe
final_df = pd.concat([df[['Drug']], maccs_df], axis=1)
final_df = final_df.dropna()

# Print the first 5 rows
print(final_df.head(5))

# Save to CSV for further analysis
final_df.to_csv("MACCS/MACCS_Keys_Output.csv", index=False)