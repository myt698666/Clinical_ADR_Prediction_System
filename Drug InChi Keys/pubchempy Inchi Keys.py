import csv
from pubchempy import get_compounds

drug_names = ['Citalopram', 'Dapoxetine', 'Escitalopram', 'fluoxetine', 'fluvoxamine', 'paroxetine', 'sertraline', 'vortioxetine']  # Replace with your list
output_file = "Drug Inchi Keys/SSRIs Inhibitors Inchi.csv"

# Open the file in write mode
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    # Write the header

    # Process each drug and write its InChI Key
    writer.writerow(["Drug", "InChI Key", "smiles"])        
    for drug in drug_names:
        compounds = get_compounds(drug, 'name')
        if compounds:
            print(f"Drug: {drug}, InChI Key: {compounds[0].inchikey}, SMILES: {compounds[0].canonical_smiles}")
            writer.writerow([drug, compounds[0].inchikey, compounds[0].canonical_smiles])
        else:
            print(f"Drug: {drug}, No InChI Key found.")
            writer.writerow([drug, "No InChI Key found"])

print(f"Results saved to {output_file}")
