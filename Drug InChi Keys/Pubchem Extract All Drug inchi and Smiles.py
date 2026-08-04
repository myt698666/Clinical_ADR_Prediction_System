import csv
from pubchempy import get_compounds

# File paths
input_file = "extracted_links_and_names.csv"  # Your input CSV file
output_file = "Drug Inchi Keys/All_drug_Inchi_and_smiles.csv"

# Open the input file and read the drug names from the "Drug Names" column
drug_names = []
with open(input_file, mode="r", newline="", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        drug_names.append(row["Drug Name"])  # Assuming the column name is "Drug Names"

# Open the output file in write mode
with open(output_file, mode="w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    # Write the header including PubChem CID
    writer.writerow(["Drug", "InChI Key", "SMILES", "PubChem CID"])
    
    # Process each drug and write its InChI Key, SMILES, and PubChem CID
    for drug in drug_names:
        compounds = get_compounds(drug, 'name')
        if compounds:
            pubchem_cid = compounds[0].cid if hasattr(compounds[0], 'cid') else "No PubChem CID found"
            print(f"Drug: {drug}, InChI Key: {compounds[0].inchikey}, SMILES: {compounds[0].canonical_smiles}, PubChem CID: {pubchem_cid}")
            writer.writerow([drug, compounds[0].inchikey, compounds[0].canonical_smiles, pubchem_cid])
        else:
            print(f"Drug: {drug}, No InChI Key found.")
            writer.writerow([drug, "No InChI Key found", "No SMILES found", "No PubChem CID found"])

print(f"Results saved to {output_file}")
