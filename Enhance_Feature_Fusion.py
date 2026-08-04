import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import os

print("Step 1/4: Loading primary feature matrix (500-dimensional protein-drug interaction matrix).")

# Load the fused protein-drug interaction matrix
matrix_file = "STITCH_Identifiers/top_500_protein_drug_interaction_matrix.csv"
protein_matrix = pd.read_csv(matrix_file)

print("Step 2/4: Reading chemical SMILES representations.")

# Load the drug SMILES structures
smiles_file = "Drug InChi Keys/All_drug_Inchi_and_smiles.csv"
smiles_df = pd.read_csv(smiles_file)

# Merge protein matrix with SMILES structures for validated drug compounds
elite_drugs_smiles = pd.merge(
    protein_matrix[['Matched Drug']],
    smiles_df[['Drug', 'SMILES']],
    left_on='Matched Drug',
    right_on='Drug',
    how='inner'
)

def get_morgan_fingerprint(smiles_string):
    """
    Converts SMILES string to a 2048-bit Morgan Fingerprint (ECFP4 equivalent).
    Returns a zero-vector if the SMILES string is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(str(smiles_string))
        if mol:
            # Radius 2, 2048 bits corresponding to ECFP4 fingerprinting
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            return list(fp)
        else:
            return [0] * 2048
    except Exception:
        return [0] * 2048

print("Step 3/4: Initializing RDKit engine for 2048-dimensional molecular fingerprint generation.")

# Generate Morgan fingerprints for chemical structures
fingerprints = elite_drugs_smiles['SMILES'].apply(get_morgan_fingerprint)

# Construct fingerprint dataframe
fp_df = pd.DataFrame(fingerprints.tolist(), columns=[f'ChemFp_{i}' for i in range(2048)])
fp_df.insert(0, 'Matched Drug', elite_drugs_smiles['Matched Drug'])

print("Step 4/4: Executing multi-modal feature fusion.")

# Merge protein target affinity matrix with chemical fingerprint matrix
final_fusion_matrix = pd.merge(protein_matrix, fp_df, on='Matched Drug', how='inner')

# Export the final high-dimensional feature matrix
output_path = "STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv"
final_fusion_matrix.to_csv(output_path, index=False)

print("\nProcess completed successfully.")
print(f"Fused matrix saved to: {output_path}")
print(f"Final feature dimension: {final_fusion_matrix.shape[1] - 1} dimensions (500 protein targets + 2048 chemical structural features).")