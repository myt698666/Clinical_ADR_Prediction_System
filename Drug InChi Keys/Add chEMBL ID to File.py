import csv
import time
from chembl_webresource_client.new_client import new_client

# File paths
input_file = "Drug Inchi Keys/All_drug_Inchi_and_smiles.csv"  # Your original CSV file
output_file = "Drug Inchi Keys/All_drug_Inchi_and_smiles_with_chembl.csv"  # Output CSV file

# Connect to the ChEMBL database
molecule = new_client.molecule

# Open the input file and read the drug InChI Keys and other columns
with open(input_file, mode="r", newline="", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    rows = list(reader)  # Read all rows

    # Add ChEMBL ID and ATC Code to each row
    for idx, row in enumerate(rows):
        inchi_key = row["InChI Key"]  # Get the InChI Key from the row
        
        # Print status update for the current row
        print(f"Processing row {idx + 1} for drug: {row['Drug']} with InChI Key: {inchi_key}")
        
        # Retry logic
        retries = 3  # Number of retry attempts
        for attempt in range(retries):
            try:
                # Search for the molecule in ChEMBL by InChI Key
                results = molecule.filter(inchi_key=inchi_key)
                
                if results:
                    # Extract the first result (in case multiple results are returned)
                    chembl_data = results[0]
                    
                    # Get the ChEMBL ID
                    chembl_id = chembl_data['molecule_chembl_id']
                    
                    # Get the ATC Code(s) (if available)
                    atc_codes = chembl_data.get('atc_classifications', [])
                    atc_code = None
                    if atc_codes:
                        # Take the first ATC code if available
                        atc_code = atc_codes[0]['code']
                    
                    # Add ChEMBL ID and ATC Code to the row
                    row["ChEMBL ID"] = chembl_id
                    row["ATC Code"] = atc_code if atc_code else "No ATC Code found"
                else:
                    # If no result is found, add placeholder values
                    row["ChEMBL ID"] = "No ChEMBL ID found"
                    row["ATC Code"] = "No ATC Code found"
                
                break  # Exit the retry loop if successful

            except Exception as e:
                # Print a more concise error message
                print(f"Error on attempt {attempt + 1} for drug {row['Drug']} (InChI Key: {inchi_key}): {str(e)}")
                
                if attempt < retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)  # Wait for 5 seconds before retrying
                else:
                    row["ChEMBL ID"] = "Error after retries"
                    row["ATC Code"] = "Error after retries"
                    print(f"Failed to retrieve ChEMBL ID and ATC Code for drug: {row['Drug']}")

# Write the updated data back into the original CSV file or a new one
with open(output_file, mode="w", newline="", encoding="utf-8") as outfile:
    fieldnames = reader.fieldnames + ["ChEMBL ID", "ATC Code"]  # Add new columns to the fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    # Write the header
    writer.writeheader()
    
    # Write the updated rows
    writer.writerows(rows)

print(f"Results saved to {output_file}")
