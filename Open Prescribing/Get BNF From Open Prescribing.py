import os 
import requests
import csv
import time

# Path to the CSV file containing drug names
input_file_path = 'Drug Inchi Keys/All_drug_Inchi_and_smiles.csv'

# Output file where all the data will be stored
output_file_path = 'Open Prescribing/bnf/all_chemical_bnf_codes.csv'

# Read drug names from the 'Drug' column of the CSV file
drug_names = []
with open(input_file_path, 'r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        drug_names.append(row['Drug'])  # Assuming the 'Drug' column contains the drug names

# Base API endpoint
base_url = "https://openprescribing.net/api/1.0/bnf_code/?q={drug_name}&format=json"

# Open the output file once at the beginning, outside of the loop
with open(output_file_path, 'w', newline='') as outfile:
    csvwriter = csv.writer(outfile)
    written_headers = False  # To check if headers are already written

    # Iterate through each drug name from the CSV file
    for drug_name in drug_names:
        # Construct the API URL for the current drug
        url = base_url.format(drug_name=drug_name)
        
        # Make the GET request
        response = requests.get(url)
        
        if response.status_code == 200:
            # Parse response JSON and reset `items` for each iteration
            items = response.json()
            
            if items:
                # Filter only items with type 'chemical' and remove the 'type' field
                chemical_items = [
                    {key: value for key, value in item.items() if key != 'type'} 
                    for item in items if item.get("type") == "chemical"
                ]
                
                if chemical_items:
                    # Write headers only once
                    if not written_headers:
                        headers = ['Drug'] + list(chemical_items[0].keys())  # Add 'Drug' as the first column
                        csvwriter.writerow(headers)  # Write headers to the CSV
                        written_headers = True
                    
                    # Write data rows with drug name included as the first column
                    for item in chemical_items:
                        row = [drug_name] + list(item.values())  # Add drug_name as the first column
                        csvwriter.writerow(row)
                    
                    print(f"Chemical data for {drug_name} has been successfully written to {output_file_path}.")
                else:
                    print(f"No chemical data found for {drug_name}.")
            else:
                print(f"No data found for {drug_name}.")
        else:
            print(f"Failed to fetch data for {drug_name}, status code: {response.status_code}")
        
        # Add a small delay between iterations to avoid rate limits
        time.sleep(0.01)

print(f"All chemical data has been saved to {output_file_path}.")
