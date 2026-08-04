import pandas as pd
import pubchempy as pcp
import time
import os
from tqdm import tqdm

# Configuration for data ingestion and output paths
input_data_primary = "All_filtered_chemicals_file.tsv"
input_data_secondary = "Drug InChi Keys/SSRIs Inhibitors Inchi.csv"
output_directory = "STITCH_Identifiers"
output_file_path = os.path.join(output_directory, "matched_chemicals.csv")

os.makedirs(output_directory, exist_ok=True)

# Load primary datasets
df_primary = pd.read_csv(input_data_primary, sep="\t")
df_secondary = pd.read_csv(input_data_secondary)

print("Initializing PubChem API search engine for drug name standardization.")

# Data deduplication: extracting unique chemical identifiers from both sources
unique_names_primary = df_primary['name'].dropna().unique()
unique_names_secondary = df_secondary['Drug'].dropna().unique()
all_unique_names = list(set(unique_names_primary) | set(unique_names_secondary))

print(f"Total unique drug identifiers for querying: {len(all_unique_names)}")

# Dictionary to store standardized CID mappings
cid_mapping_registry = {}

def fetch_standard_cid(drug_name):
    """
    Standardizes drug names to PubChem CID using the PubChem API.
    """
    try:
        clean_name = str(drug_name).lower().strip()
        compounds = pcp.get_compounds(clean_name, 'name')
        if compounds:
            return compounds[0].cid
        # Rate-limiting for API compliance
        time.sleep(0.2)
    except Exception:
        return None
    return None

# Execute API batch queries
print("Executing batch retrieval of chemical identifiers...")
for name in tqdm(all_unique_names):
    cid = fetch_standard_cid(name)
    if cid is not None:
        cid_mapping_registry[name] = cid

# Identifier mapping and dataset consolidation
print("Mapping CIDs back to original datasets and performing inner join.")

df_primary['CID'] = df_primary['name'].map(cid_mapping_registry)
df_secondary['CID'] = df_secondary['Drug'].map(cid_mapping_registry)

# Filtering invalid records and merging datasets
df_primary_valid = df_primary.dropna(subset=['CID'])
df_secondary_valid = df_secondary.dropna(subset=['CID'])

consolidated_df = pd.merge(df_primary_valid, df_secondary_valid, on='CID', how='inner')
final_result = consolidated_df[['chemical', 'Drug']]

# Export results
final_result.to_csv(output_file_path, index=False)

print("\nStandardization process completed.")
print(f"Total valid matched records: {len(final_result)}")
print(f"Resulting mapping matrix exported to: {output_file_path}")