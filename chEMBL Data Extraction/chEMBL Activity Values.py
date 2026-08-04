import pandas as pd
import json
import os
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.http_errors import HttpApplicationError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Initialize ChEMBL clients
molecule = new_client.molecule
activity = new_client.activity

# Read InChI keys from the CSV file
input_file = "Drug Inchi Keys/All_drug_Inchi_and_smiles.csv"
progress_file = "progress.json"
output_file = "Drug Activity/chEMBL_activity_values.csv"
intermediate_file = "Drug Activity/intermediate_results.csv"

# Load input data
inchi_keys_df = pd.read_csv(input_file)

# Initialize results list
all_results = []

# Load progress if available
start_inchi_key = None
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        progress_data = json.load(f)
        start_inchi_key = progress_data.get('last_processed_key')

# Define a retry decorator for handling temporary API failures
@retry(
    retry=retry_if_exception_type(HttpApplicationError),  # Retry on specific exceptions
    wait=wait_exponential(multiplier=1, min=4, max=60),   # Exponential backoff
    stop=stop_after_attempt(5),                           # Stop after 5 attempts
    reraise=True                                          # Raise exception if retries fail
)
def fetch_bioactivities(chembl_id):
    """
    Fetches bioactivity data for a given ChEMBL ID with retry logic.
    """
    return activity.filter(molecule_chembl_id=chembl_id)

# Process each InChI Key
for i, inchi_key in enumerate(inchi_keys_df['InChI Key']):
    if inchi_key == start_inchi_key:
        continue  # Skip already processed

    # Search for the molecule by InChI Key
    result = molecule.filter(molecule_structures__standard_inchi_key=inchi_key).only(['molecule_chembl_id', 'pref_name'])
    
    if result:
        # Get the ChEMBL ID and preferred name
        chembl_id = result[0]['molecule_chembl_id']
        molecule_name = result[0].get('pref_name', 'Unknown')
        
        # Print the preferred name to track progress
        print(f"Processing: {molecule_name} (InChI Key: {inchi_key})")
        
        try:
            # Fetch all bioactivity data for the molecule using retry logic
            bioactivities = fetch_bioactivities(chembl_id)
        except HttpApplicationError as e:
            print(f"Failed to fetch bioactivities for {molecule_name} (InChI Key: {inchi_key}): {e}")
            continue  # Skip this molecule and proceed to the next
        
        # Define the standard types of interest
        desired_standard_types = {'AC50', 'EC50', 'IC50', 'ED50', 'Ki'}
        
        # Filter data based on the standard type and standard value
        filtered_data = [
            {
                "Molecule Name": molecule_name,
                "Molecule ChEMBL ID": chembl_id,
                "Target ChEMBL ID": b.get('target_chembl_id'),
                "Target Name": b.get('target_pref_name', 'Unknown'),
                "Activity ID": b.get("activity_id"),
                "Assay ChEMBL ID": b.get("assay_chembl_id"),
                "Standard Type": b.get("standard_type"),
                "Standard Value": float(b.get("standard_value", 0)),  # Ensure value is numeric
                "Standard Units": b.get("standard_units"),
                "Reference": b.get("document_chembl_id"),
            }
            for b in bioactivities
            if b.get('standard_type') in desired_standard_types and b.get('standard_value') is not None
        ]
        
        all_results.extend(filtered_data)
    else:
        print(f"No ChEMBL ID found for InChI Key: {inchi_key}")
    
    # Save intermediate results to CSV
    pd.DataFrame(all_results).to_csv(intermediate_file, index=False)
    
    # Save progress after each iteration
    with open(progress_file, 'w') as f:
        json.dump({'last_processed_key': inchi_key}, f)

# Save final results to CSV
pd.DataFrame(all_results).to_csv(output_file, index=False)

print(f"Processing complete. Results saved to {output_file}.")
