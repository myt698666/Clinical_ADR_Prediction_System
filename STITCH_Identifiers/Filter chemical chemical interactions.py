import pandas as pd
import os

# File paths
# --- 强力注入我们的完全体数据 ---
if os.path.exists("matched_chemicals_2768.csv"):
    matched_file = "matched_chemicals_2768.csv"
else:
    matched_file = "STITCH_Identifiers/matched_chemicals_2768.csv"
chemical_chemical_file = "chemical_chemical.links.v5.0.tsv"
output_folder = "STITCH_Identifiers"
output_file = os.path.join(output_folder, "filtered_chemical_chemical_interactions.csv")

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load the files
matched_df = pd.read_csv(matched_file)
chemical_chemical_df = pd.read_csv(chemical_chemical_file, sep="\t")  # Assuming TSV format

# Normalise the 'chemical' columns (lowercase, trimmed)
matched_df['chemical_normalized'] = matched_df['chemical'].str.lower().str.strip()
chemical_chemical_df['chemical1_normalized'] = chemical_chemical_df['chemical1'].str.lower().str.strip()
chemical_chemical_df['chemical2_normalized'] = chemical_chemical_df['chemical2'].str.lower().str.strip()

# Step 1: Filter chemical_chemical_df based on the 'chemical1' and 'chemical2' columns in matched_df
filtered_df = chemical_chemical_df[
    (chemical_chemical_df['chemical1_normalized'].isin(matched_df['chemical_normalized'])) |
    (chemical_chemical_df['chemical2_normalized'].isin(matched_df['chemical_normalized']))
]

# Step 2: Add the 'Drug' column for both `chemical1` and `chemical2`
# Merge for `chemical1`
filtered_df = pd.merge(
    filtered_df,
    matched_df[['chemical_normalized', 'Drug']],
    left_on='chemical1_normalized',
    right_on='chemical_normalized',
    how='inner'
)

filtered_df.rename(columns={'Drug': 'Drug_from_chemical1'}, inplace=True)

# Merge for `chemical2`
filtered_df = pd.merge(
    filtered_df,
    matched_df[['chemical_normalized', 'Drug']],
    left_on='chemical2_normalized',
    right_on='chemical_normalized',
    how='left'
)

filtered_df.rename(columns={'Drug': 'Drug_from_chemical2'}, inplace=True)

# Step 3: Keep only relevant columns and filter rows
final_result = filtered_df[['chemical1', 'Drug_from_chemical1', 'chemical2', 'Drug_from_chemical2', 'textmining']]

# Filter 'textmining' to include only values greater than 400
final_result = final_result[final_result['textmining'] > 400]

# Divide the 'textmining' values by 1000 to normalise
final_result['textmining'] = final_result['textmining'] / 1000

# Rename columns for clarity
final_result.rename(columns={
    'chemical1': 'Chemical 1',
    'Drug_from_chemical1': 'Matched Drug 1',
    'chemical2': 'Chemical 2',
    'Drug_from_chemical2': 'Matched Drug 2',
    'textmining': 'Confidence'
}, inplace=True)

# Step 4: Ensure both columns are properly filled
# Replace NaN in `Matched Drug 1` and `Matched Drug 2` with empty strings
final_result['Matched Drug 1'] = final_result['Matched Drug 1'].fillna('')
final_result['Matched Drug 2'] = final_result['Matched Drug 2'].fillna('')

# Save the results to a file
final_result.to_csv(output_file, index=False)

print(f"Filtered chemical-chemical results saved to: {output_file}")
