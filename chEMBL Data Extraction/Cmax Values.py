from chembl_webresource_client.new_client import new_client
import pandas as pd
import time
import os
import random

# Initialize the molecule and activity resources
molecule = new_client.molecule
activity = new_client.activity

# File path and column name
file_path = 'Drug Inchi Keys/All_drug_Inchi_and_smiles.csv'
inchi_key_column = 'InChI Key'

# Temporary file path
temp_file_path = "Drug Activity/temp_Cmax_results.csv"

# Remove the temporary file if it already exists (to start fresh)
if os.path.exists(temp_file_path):
    os.remove(temp_file_path)

# Read the CSV file
df_inchi_keys = pd.read_csv(file_path)

# Check if the column exists
if inchi_key_column not in df_inchi_keys.columns:
    print(f"Column '{inchi_key_column}' not found in the file.")
else:
    # Retry settings
    max_retries = 3
    retry_wait_time = 5  # Time in seconds to wait between retries

    # Iterate through each InChI Key in the column
    for index, inchi_key in enumerate(df_inchi_keys[inchi_key_column], start=1):
        print(f"Processing InChI Key {index}/{len(df_inchi_keys)}: {inchi_key}")
        
        # Retry logic for querying ChEMBL
        retries = 0
        while retries < max_retries:
            try:
                # Start timing the query
                start_time = time.time()

                # Search for the molecule by InChI Key
                result = molecule.filter(molecule_structures__standard_inchi_key=inchi_key).only(['molecule_chembl_id', 'pref_name'])

                if result:
                    # Get the ChEMBL ID and preferred name
                    chembl_id = result[0]['molecule_chembl_id']
                    molecule_name = result[0].get('pref_name', 'Unknown')
                    print(f"Found ChEMBL ID: {chembl_id}, Molecule Name: {molecule_name}")
                    
                    # Fetch all bioactivity data for the molecule
                    bioactivities = activity.filter(molecule_chembl_id=chembl_id)
                    
                    # Define the desired standard type: Cmax
                    desired_standard_types = {'Cmax'}
                    
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
                            "Standard Value": float(b.get("standard_value")),  # Ensure value is numeric
                            "Standard Units": b.get("standard_units"),
                            "Reference": b.get("document_chembl_id"),
                        }
                        for b in bioactivities
                        if b.get('standard_type') in desired_standard_types and b.get('standard_value') is not None
                    ]
                    
                    # Append results to the temporary file
                    if filtered_data:
                        bioactivity_df = pd.DataFrame(filtered_data)
                        # Append to temporary file
                        bioactivity_df.to_csv(temp_file_path, mode='a', header=not os.path.exists(temp_file_path), index=False)
                        print(f"Appended results to temporary file: {temp_file_path}")
                    else:
                        print(f"No relevant bioactivity data found for {inchi_key}.")
                else:
                    print(f"No ChEMBL ID found for InChI Key: {inchi_key}")

                # End timing the query
                end_time = time.time()
                print(f"Query for {inchi_key} completed in {end_time - start_time:.2f} seconds.")
                break  # Exit loop if successful
                
            except Exception as e:
                retries += 1
                print(f"Error encountered: {e}. Retrying {retries}/{max_retries}...")
                
                if retries < max_retries:
                    wait_time = retry_wait_time * random.uniform(1, 2)  # Add randomization to wait time
                    print(f"Waiting for {wait_time:.2f} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    print(f"Max retries reached for InChI Key: {inchi_key}. Skipping.")
    
    # Rename the temporary file to the final results file
    final_file_path = "Drug Activity/final_Cmax_results.csv"
    os.rename(temp_file_path, final_file_path)
    print(f"Final results saved to: {final_file_path}")