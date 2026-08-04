import pandas as pd
import os
from fuzzywuzzy import process
from tqdm import tqdm

# Enable progress tracking for pandas operations
tqdm.pandas()

# Configuration for data ingestion and export
primary_input = "All_filtered_chemicals_file.tsv"
secondary_input = "extracted_links_and_names.csv"
output_directory = "STITCH_Identifiers"
output_file_path = os.path.join(output_directory, "matched_chemicals_hybrid.csv")

os.makedirs(output_directory, exist_ok=True)

# Load raw datasets
df_primary = pd.read_csv(primary_input, sep="\t")
df_secondary = pd.read_csv(secondary_input)

# Harmonize column naming conventions
if 'Drug Name' in df_secondary.columns:
    df_secondary = df_secondary.rename(columns={'Drug Name': 'Drug'})

# Preprocessing: standardize text case and remove whitespace for reliable matching
df_primary['name_clean'] = df_primary['name'].astype(str).str.lower().str.strip()
df_secondary['drug_clean'] = df_secondary['Drug'].astype(str).str.lower().str.strip()

print(f"Total clinical records for initial processing: {len(df_secondary)}")
print("Phase 1: Executing exact identifier matching.")

# Phase 1: Exact matching
exact_match_df = pd.merge(df_primary, df_secondary, left_on='name_clean', right_on='drug_clean', how='inner')
print(f"Phase 1 successfully matched: {len(exact_match_df)} records.")

# Phase 2: Fuzzy string matching for residual records
matched_primary_ids = exact_match_df['name_clean'].unique()
matched_secondary_ids = exact_match_df['drug_clean'].unique()

unmatched_secondary = df_secondary[~df_secondary['drug_clean'].isin(matched_secondary_ids)].copy()
unmatched_primary = df_primary[~df_primary['name_clean'].isin(matched_primary_ids)].copy()

stitch_names_pool = unmatched_primary['name_clean'].dropna().unique().tolist()

print(f"Residual records identified for fuzzy matching: {len(unmatched_secondary)}")
print("Phase 2: Initiating fuzzy matching engine.")

def fuzzy_match_drug(target_name, pool, threshold=88):
    """
    Computes Levenshtein-based similarity to bridge identifier discrepancies.
    """
    match_result = process.extractOne(target_name, pool)
    if match_result and match_result[1] >= threshold:
        return match_result[0]
    return None

# Perform similarity matching
unmatched_secondary['fuzzy_match_name'] = unmatched_secondary['drug_clean'].progress_apply(
    lambda x: fuzzy_match_drug(x, stitch_names_pool)
)

# Recover fuzzy-matched records
fuzzy_success_df = unmatched_secondary.dropna(subset=['fuzzy_match_name'])
fuzzy_match_df = pd.merge(
    unmatched_primary,
    fuzzy_success_df,
    left_on='name_clean',
    right_on='fuzzy_match_name',
    how='inner'
)

print(f"Phase 2 successfully recovered: {len(fuzzy_match_df)} records.")

# Final consolidation
final_merged_df = pd.concat([exact_match_df, fuzzy_match_df], ignore_index=True)
result_df = final_merged_df[['chemical', 'Drug']].drop_duplicates()

# Export standardized mapping
result_df.to_csv(output_file_path, index=False)

print("\nFinal identifier matching completed.")
print(f"Total clinical records recovered: {len(result_df)}")
print(f"Standardized mapping exported to: {output_file_path}")