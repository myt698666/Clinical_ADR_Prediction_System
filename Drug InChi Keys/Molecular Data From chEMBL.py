from chembl_webresource_client.new_client import new_client
import csv
import time

# File containing the saved InChI Keys
input_file = "Drug Inchi Keys/SSRIs Inhibitors Inchi.csv"

# Read the InChI Keys from the CSV file
inchi_keys = []
with open(input_file, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        key = row.get("InChI Key", "").strip()
        if key and key != "No InChI Key found":
            inchi_keys.append(key)

# Initialize the ChEMBL molecule client
molecule = new_client.molecule

# List to store results
results = []

for key in inchi_keys:
    try:
        print(f"Querying ChEMBL for InChI Key: {key}")
        data = molecule.filter(molecule_structures__standard_inchi_key=key)
        print(f"Data for {key}: {data}")  # Debugging line

        if data:
            mol_data = data[0]  # Assuming the first result is the most relevant
            print(f"Retrieved data for {key}: {mol_data}")
            results.append({
                "InChI Key": key,
                "ChEMBL ID": mol_data["molecule_chembl_id"],
                "Molecular Weight": mol_data.get("molecule_properties", {}).get("full_mwt", "Not Available"),
                "AlogP": mol_data.get("molecule_properties", {}).get("alogp","Not Available"),
                "Aromatic Rings": mol_data.get("molecule_properties", {}).get("aromatic_rings","Not Available"),
                "Cx LogD": mol_data.get("molecule_properties", {}).get("cx_logd","Not Available"),
                "Cx LogP": mol_data.get("molecule_properties", {}).get("cx_logp","Not Available"),
                "Cx Most Apka": mol_data.get("molecule_properties", {}).get("cx_most_apka","Not Available"),
                "Cx Most Bpka": mol_data.get("molecule_properties", {}).get("cx_most_bpka","Not Available"),
                "HBA": mol_data.get("molecule_properties", {}).get("hba","Not Available"),
                "HBD": mol_data.get("molecule_properties", {}).get("hbd","Not Available"),
                "HBA Lipinski": mol_data.get("molecule_properties", {}).get("hba_lipinski","Not Available"),
                "HBD Lipinski": mol_data.get("molecule_properties", {}).get("hbd_lipinski","Not Available"),
                "Heavy Atoms": mol_data.get("molecule_properties", {}).get("heavy_atoms","Not Available"),
                "Rotatable Bonds": mol_data.get("molecule_properties", {}).get("num_lipinski_ro5_violations","Not Available"),
                "TPSA": mol_data.get("molecule_properties", {}).get("psa","Not Available"),
                "Preferred Name": mol_data.get("pref_name"),
            })
        else:
            print(f"No data found for InChI Key: {key}")
        
        time.sleep(1)  # Pause to prevent rate-limiting
    except Exception as e:
        print(f"Error retrieving data for {key}: {e}")
        # Continue to the next key if an error occurs
        continue

# Ensure results are populated
print(f"Total results retrieved: {len(results)}")

if results:
    # Save the results to a CSV file
    output_file = "Drug InChi Keys/chembl_all_molecular_data.csv"
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Preferred Name", "InChI Key", "ChEMBL ID", "Molecular Weight", "AlogP", "Aromatic Rings", "Cx LogD", "Cx LogP", "Cx Most Apka", "Cx Most Bpka",
                "HBA", "HBD", "HBA Lipinski", "HBD Lipinski", "Heavy Atoms", "Rotatable Bonds", "TPSA"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_file}")
else:
    print("No results to save. Check if the InChI Keys are valid and retrievable.")