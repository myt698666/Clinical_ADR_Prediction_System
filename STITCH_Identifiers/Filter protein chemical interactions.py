import pandas as pd
import os

# File paths
matched_file = os.path.join("STITCH_Identifiers", "All_Matched_Chemicals.csv")
protein_chemical_file = "9606.protein_chemical.links.v5.0.tsv"
protein_results_file = "9606.protein.info.v12.0.csv"
output_folder = "STITCH_Identifiers"
output_file = os.path.join(output_folder, "filtered_protein_chemical_interactions.csv")

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

import os

# --- 强力注入我们的完全体数据，并自动适配相对路径 ---
if os.path.exists("matched_chemicals_2768.csv"):
    matched_file = "matched_chemicals_2768.csv"
else:
    matched_file = "STITCH_Identifiers/matched_chemicals_2768.csv"

# Load the files
matched_df = pd.read_csv(matched_file)
protein_chemical_df = pd.read_csv(protein_chemical_file, sep="\t")  # Assuming TSV format
protein_results_df = pd.read_csv(protein_results_file)

# Normalize the 'chemical' columns (lowercase, trimmed)
matched_df['chemical_normalized'] = matched_df['chemical'].str.lower().str.strip()
protein_chemical_df['chemical_normalized'] = protein_chemical_df['chemical'].str.lower().str.strip()

# Step 1: Filter protein_chemical_df based on the 'chemical' column in matched_df
filtered_df = protein_chemical_df[
    protein_chemical_df['chemical_normalized'].isin(matched_df['chemical_normalized'])
]

# Step 2: Match the 'protein' column in filtered_df with the 'STRING_ID' column in protein_results_df
merged_df = pd.merge(
    filtered_df,
    protein_results_df,
    left_on='protein',
    right_on='#string_protein_id',
    how='inner'
)

# Step 3: Add the 'Drug' column from matched_df based on the chemical normalization
final_result = pd.merge(
    merged_df,
    matched_df[['chemical_normalized', 'Drug']],
    on='chemical_normalized',
    how='inner'
)

# Step 4: Add the 'score' column from protein_chemical_df
final_result = final_result[['chemical', 'Drug', 'preferred_name', 'annotation', 'combined_score']]

# Filter 'combined_score' to include only values greater than 400
final_result = final_result[final_result['combined_score'] > 400]

# Divide the 'combined_score' values by 1000
final_result['combined_score'] = final_result['combined_score'] / 1000

# Rename columns for clarity
final_result.rename(columns={
    'chemical': 'Chemical',
    'Drug': 'Matched Drug',
    'preferred_name': 'Protein Name',
    'combined_score': 'Confidence'
}, inplace=True)


# Save the results to a file
final_result.to_csv(output_file, index=False)

print(f"Filtered protein-chemical results saved to: {output_file}")
