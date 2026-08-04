import os
import re
import requests
import zipfile
import csv

# Function to sanitize the filename by replacing disallowed characters with underscores
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# Define the base URL for the alternate link
base_url = "https://info.mhra.gov.uk/drug-analysis-profiles/data"

# Directory where you want to save the extracted files
extract_dir = 'All_ADR_Event_Files'

# Ensure the extraction directory exists
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

# Read the CSV file to get the drug names and their corresponding links
input_csv_file = 'extracted_zipped_links_and_names.csv'

# Read the CSV file
with open(input_csv_file, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    drug_links = [
        {'Drug Name': row['Drug Name'], 'Zipped Link': row['Zipped Link']}
        for row in csv_reader
    ]

# Process each drug name and its link
for item in drug_links:
    drug_name = item['Drug Name']
    alternate_link = item['Zipped Link']

    sanitzed_drug_name = sanitize_filename(drug_name)
    # Download the ZIP file
    response = requests.get(alternate_link)
    
    # Check if the download was successful
    if response.status_code == 200:
        # Construct the ZIP file name
        zip_filename = os.path.join(extract_dir, f"{sanitzed_drug_name}_{alternate_link.split('/')[-1]}")
        
        # Save the downloaded zip file to disk
        with open(zip_filename, 'wb') as zip_file:
            zip_file.write(response.content)
        
        # Extract the ZIP file
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            # Define the allowed file patterns
            allowed_patterns = ('_case.csv', '_drug.csv', '_event.csv')
            
            # Extract and rename only the allowed files
            for file_name in zip_ref.namelist():
                if file_name.endswith(allowed_patterns):
                    extracted_path = zip_ref.extract(file_name, extract_dir)
                    
                    # Simplify the filename by prepending the drug name
                    suffix = file_name.split('_')[-1]  # Get the suffix, e.g., 'case.csv'
                    # Use this function when constructing the new filename
                    new_name = f"{sanitzed_drug_name}_{suffix}"
                    new_file_path = os.path.join(extract_dir, new_name)
                    os.rename(extracted_path, new_file_path)
                    print(f"Extracted and renamed {file_name} to {new_name}")
        
        # Delete the ZIP file after extraction
        os.remove(zip_filename)
        print(f"Deleted ZIP file: {zip_filename}")
    
    else:
        print(f"Failed to download {drug_name} from {alternate_link}")

print("Processing completed.")