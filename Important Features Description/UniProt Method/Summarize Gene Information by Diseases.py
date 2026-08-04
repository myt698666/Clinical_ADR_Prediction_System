import pandas as pd
import glob
import re
import os

# Define input and output folder paths
input_folder = "Important Features Description/Model Descriptions/"
output_folder = "Important Features Description/Disease Summary/"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Adjust the file extension if needed (here we assume CSV files)
file_pattern = os.path.join(input_folder, "*.csv")
files = glob.glob(file_pattern)

def extract_diseases(cell):
    """
    Extracts disease names from a cell in the 'Involvement in disease' column.
    Expected format for each disease entry is something like:
    "DISEASE: DiseaseName [MIM:123456]: Description ...".
    If multiple entries exist, they should be separated by semicolons.
    """
    if pd.isna(cell) or cell.strip() == "":
        return []
    
    # Split the cell on semicolons (each entry is one disease record)
    entries = [entry.strip() for entry in cell.split(";")]
    diseases = []
    
    # Loop through each disease entry
    for entry in entries:
        # If the entry starts with "DISEASE:" try to extract the disease name.
        match = re.search(r"DISEASE:\s*(.*?)\s*:", entry)
        if match:
            disease_name = match.group(1)
            diseases.append(disease_name)
        else:
            # If the expected pattern isn't found, include the entire entry
            diseases.append(entry)
    return diseases

# Process each file individually
for file in files:
    # Read each file into a DataFrame (adjust read_csv if your file has a different delimiter or encoding)
    df = pd.read_csv(file)
    
    # Check if the necessary columns exist
    if "Involvement in disease" not in df.columns or "Gene" not in df.columns:
        print(f"Skipping {file}: required columns not found.")
        continue
    
    # Extract disease names into a new column
    df["disease_list"] = df["Involvement in disease"].apply(extract_diseases)
    
    # Group by Gene and aggregate all disease lists into a unique set per gene.
    disease_by_gene = (
        df.groupby("Gene")["disease_list"]
        .apply(lambda lists: set(sum(lists, [])))
        .reset_index()
    )
    
    # Save the grouped results to a new CSV file in the output folder using the same file name.
    base_name = os.path.basename(file)
    output_file_path = os.path.join(output_folder, base_name)
    disease_by_gene.to_csv(output_file_path, index=False)
    
    print(f"Processed and saved: {output_file_path}")
