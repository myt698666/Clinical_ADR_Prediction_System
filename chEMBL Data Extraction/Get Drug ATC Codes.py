import pandas as pd
from chembl_webresource_client.new_client import new_client
import os

# Initialize ChEMBL client
molecule = new_client.molecule

def get_atc_codes_for_inchi(inchi_key):
    """
    Fetch ATC codes for a drug using its InChI Key from ChEMBL.
    
    Args:
        inchi_key (str): The InChI Key of the drug.
    
    Returns:
        dict: A dictionary with the InChI Key, ATC codes, and ChEMBL ID.
    """
    try:
        # Search by InChI Key
        results = molecule.filter(molecule_structures__standard_inchi_key=inchi_key)
        if results:
            for result in results:
                if "atc_classifications" in result:
                    return {
                        "inchi_key": inchi_key,
                        "atc_codes": result.get("atc_classifications", []),
                        "chembl_id": result.get("molecule_chembl_id", "")
                    }
        return {"inchi_key": inchi_key, "atc_codes": [], "chembl_id": None}
    except Exception as e:
        print(f"Error fetching ATC codes for {inchi_key}: {e}")
        return {"inchi_key": inchi_key, "atc_codes": [], "chembl_id": None}

# File paths
input_file = "Drug InChi Keys/All_drug_Inchi_and_smiles.csv"
output_file = "Drug InChi Keys/Drug_ATC_Codes.csv"
temp_file = "Drug InChi Keys/temp_progress.csv"

# Step 1: Load the CSV file
data = pd.read_csv(input_file)

# Step 2: Filter out rows where 'InChI Key' is 'No InChI Key found'
filtered_data = data[data['InChI Key'] != 'No InChI Key found']

# Step 3: Fetch ATC codes for each valid InChI Key
results = []
for index, row in filtered_data.iterrows():
    drug_name = row['Drug']
    inchi_key = row['InChI Key']
    
    # Print progress
    print(f"Processing: Drug = {drug_name}, InChI Key = {inchi_key}")
    
    # Fetch ATC codes
    result = get_atc_codes_for_inchi(inchi_key)
    result["drug_name"] = drug_name
    results.append(result)
    
    # Save progress to a temporary file
    temp_df = pd.DataFrame(results)
    temp_df.to_csv(temp_file, index=False)
    
# Step 4: Save final results to output file
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f"ATC codes saved to {output_file}")

# Step 5: Delete the temporary file
if os.path.exists(temp_file):
    os.remove(temp_file)
    print(f"Temporary file {temp_file} deleted.")
