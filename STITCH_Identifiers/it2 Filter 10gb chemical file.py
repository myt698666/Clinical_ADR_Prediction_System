import pandas as pd
import os

# File paths
large_tsv_path = "chemicals.v5.0.tsv"
smiles_list_csv_path = "Drugs and keys/Updated Drug Master Keys.csv"
output_file_path = "IT2_All_filtered_chemicals_file.tsv"

# Load the SMILES strings
drug_list = pd.read_csv(smiles_list_csv_path, usecols=["SMILES"])
drug_names = set(drug_list["SMILES"].str.strip())  # Use a set for faster lookup

# Parameters for chunking
chunk_size = 100000  # Number of rows per chunk
total_chunks = 1163  # Known number of chunks

# Process the file in chunks
filtered_rows = []
chunks_processed = 0

for chunk in pd.read_csv(large_tsv_path, sep="\t", chunksize=chunk_size):
    chunks_processed += 1

    # Apply exact match filtering using .isin
    exact_match_filter = chunk["SMILES_string"].isin(drug_names)
    filtered_chunk = chunk[exact_match_filter]
    filtered_rows.append(filtered_chunk)

    # Progress feedback
    print(f"Processed chunk {chunks_processed} of {total_chunks}. Remaining: {total_chunks - chunks_processed}")

# Combine and save filtered rows
filtered_data = pd.concat(filtered_rows)
filtered_data.to_csv(output_file_path, sep="\t", index=False)

print(f"Filtered data saved to {output_file_path}")