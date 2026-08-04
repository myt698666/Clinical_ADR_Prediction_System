import pandas as pd
from chembl_webresource_client.new_client import new_client
import time
import os

# Load the file containing the original ChEMBL keys
file_path = 'Drugs and keys/Drug Master Keys.csv'
data = pd.read_csv(file_path)

# Ensure the 'chembl_id' column exists
if 'chembl_id' not in data.columns:
    raise ValueError("'chembl_id' column not found in the file")

# Drop rows where any value is blank
data = data.dropna(subset=['chembl_id'])

# Reset the DataFrame index to avoid KeyError
data = data.reset_index(drop=True)

# Create a ChEMBL client
similarity = new_client.similarity

# File to log progress of accumulated data (success/failure)
progress_log_path = 'Drugs and keys/Progress Log.csv'

# Create or overwrite the progress log file
with open(progress_log_path, 'w') as log_file:
    log_file.write("Drug,chembl_id,Similar drug,Similar chembl id,status\n")  # CSV header

    # Create temporary file to accumulate data
    temp_file_path = 'Drugs and keys/Temporary Drug Master Keys.csv'
    
    # Initialize the temporary file with headers
    with open(temp_file_path, 'w') as temp_file:
        temp_file.write("Drug,chembl_id,Similar drug,Similar chembl id,SMILES,InChI Key,ATC Classifications\n")  # Temporary file header

    # Loop through each ChEMBL ID and query for similar drugs
    for idx, chembl_id in enumerate(data['chembl_id']):
        drug_name = data.loc[idx, 'Drug']  # Get the drug name
        print(f"Processing {chembl_id} ({idx + 1}/{len(data['chembl_id'])})...")  # Print current drug

        retries = 3  # Set maximum retry attempts
        success = False
        attempt = 0

        while not success and attempt < retries:
            try:
                # Get similar drugs with a threshold of 90
                results = similarity.filter(chembl_id=chembl_id, similarity=90)

                # Loop through each result and add a new row for each similar drug
                for drug in results:
                    similar_drug_name = drug.get('pref_name', 'Unknown')  # Default to 'Unknown' if no name
                    similar_chembl_id = drug.get('molecule_chembl_id', 'Unknown')  # Default to 'Unknown' if no ID
                    
                    # Extract SMILES and InChI Key from the 'molecule_structures' field
                    molecule_structures = drug.get('molecule_structures', {})
                    smiles = molecule_structures.get('canonical_smiles', 'Unknown')  # SMILES field
                    inchi_key = molecule_structures.get('standard_inchi_key', 'Unknown')  # InChI Key field
                    
                    # Extract ATC classifications (as a list)
                    atc_classifications = drug.get('atc_classifications', [])
                    atc_classifications_str = "; ".join(atc_classifications) if atc_classifications else "Unknown"

                    # Create the row to be written to the temporary CSV file
                    new_row = {
                        'Drug': drug_name,  # Original drug's name
                        'chembl_id': chembl_id,  # Original drug's chembl ID
                        'Similar drug': similar_drug_name,  # Similar drug's name
                        'Similar chembl id': similar_chembl_id,  # Similar drug's ChEMBL ID
                        'SMILES': smiles,  # SMILES of similar drug
                        'InChI Key': inchi_key,  # InChI Key of similar drug
                        'ATC Classifications': atc_classifications_str  # ATC classifications
                    }

                    # Write the new row immediately to the temporary CSV file
                    new_data = pd.DataFrame([new_row])
                    new_data.to_csv(temp_file_path, mode='a', header=False, index=False)

                # Log success for the current drug
                log_message = f"{drug_name},{chembl_id},Success\n"
                log_file.write(log_message)
                log_file.flush()  # Write log immediately
                success = True
                print(f"Request for {chembl_id} completed successfully.\n")

            except Exception as e:
                # Log failure and retry
                attempt += 1
                log_message = f"{drug_name},{chembl_id},Failed (Attempt {attempt})\n"
                log_file.write(log_message)
                log_file.flush()  # Write log immediately
                print(f"Error fetching similar drugs for {chembl_id} (Attempt {attempt}): {e}")

                if attempt < retries:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)  # Wait 5 seconds before retrying
                else:
                    print(f"Max retries reached for {chembl_id}. Moving to next drug.")

# Save the combined dataset to a new file
final_output_path = 'Drugs and keys/Updated Drug Master Keys.csv'
# Check if the temporary file exists
if not os.path.exists(temp_file_path):
    print("Temporary file not found. No data to finalise.")
else:
    # Delete the existing final file if it exists
    if os.path.exists(final_output_path):
        os.remove(final_output_path)
        print(f"Existing final file '{final_output_path}' deleted.")

    # Rename the temporary file to the final output file
    os.rename(temp_file_path, final_output_path)
    print(f"Temporary file renamed to '{final_output_path}'. Final output saved.")
