import os
import re
import requests
import zipfile
import csv

# Define the base URL for the alternate link
base_url = "https://info.mhra.gov.uk/drug-analysis-profiles/data"

# List of drug names you want to extract
drug_list = [
    'Citalopram', 'Dapoxetine', 'Escitalopram', 'Fluoxetine', 'Fluvoxamine', 'Paroxetine', 'Sertraline', 'Vortioxetine'
    ]

# Directory where you want to save the extracted files
extract_dir = 'extracted_files'

# Ensure the extraction directory exists
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

# Path to your HTML file
file_path = 'Yellow Card Drug Infomation.txt'

# Read the content of the file
with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Define the regex pattern to match the links and drug names
pattern = r'<a href="(dap\.html\?drug=\.\/UK_EXTERNAL\/NONCOMBINED[^\s]*?)"[^>]*>([^<]+)</a>'

# Use re.findall to find all matches in the HTML content
matches = re.findall(pattern, html_content)

# Prepare the output for CSV
with open('filtered_zipped_links_and_names.csv', 'w', newline='', encoding='utf-8') as output_file:
    csv_writer = csv.writer(output_file)
    
    # Write the header row
    csv_writer.writerow(['Zipped Link', 'Drug Name'])
    
    # Process each match to create the alternate link, download and extract files
    for link, drug_name in matches:
        # Only process if the drug name is in the provided list
        if drug_name in drug_list:
            # Extract the part of the link we want, starting from /UK_EXTERNAL/NONCOMBINED/ to the end of the .zip part
            match = re.search(r'\/UK_EXTERNAL\/NONCOMBINED\/([^\s]+\.zip)', link)
            
            if match:
                # Extract the path from the URL (e.g., /UK_EXTERNAL/NONCOMBINED/UK_NON_000225104602.zip)
                zip_path = match.group(0)
                
                # Create the alternate link by combining the base URL with the extracted zip path
                alternate_link = base_url + zip_path
                
                # Write the new alternate link and drug name to the CSV
                csv_writer.writerow([drug_name,alternate_link])
                
                # Download the ZIP file
                response = requests.get(alternate_link)
                
                # Check if the download was successful
                if response.status_code == 200:
                    zip_filename = os.path.join(extract_dir, f"{drug_name}_{zip_path.split('/')[-1]}")
                    
                    # Save the downloaded zip file to disk
                    with open(zip_filename, 'wb') as zip_file:
                        zip_file.write(response.content)
                    
                    # Extract the ZIP file
                    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                        # Extract all the contents into a temporary directory
                        temp_dir = os.path.join(extract_dir, 'temp')
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        zip_ref.extractall(temp_dir)
                    
                    # Rename the extracted files by prepending the drug name
                    for root, dirs, files in os.walk(temp_dir):
                        for file_name in files:
                            # Extract the part of the file name after the last underscore
                            base_name = file_name.rsplit('_', 1)[-1]
                            
                            # Construct the new file name
                            new_name = f"{drug_name}_{base_name}"
                            old_file_path = os.path.join(root, file_name)
                            new_file_path = os.path.join(extract_dir, new_name)
                            
                            # Rename the file
                            os.rename(old_file_path, new_file_path)
                            print(f"Extracted and renamed {file_name} to {new_name}")
                    
                    # Remove the temporary directory and ZIP file after processing
                    for root, dirs, files in os.walk(temp_dir, topdown=False):
                        for file in files:
                            os.remove(os.path.join(root, file))
                        for dir in dirs:
                            os.rmdir(os.path.join(root, dir))
                    os.rmdir(temp_dir)
                    
                    os.remove(zip_filename)  # Delete the ZIP file after extraction
                    print(f"Deleted ZIP file: {zip_filename}")
                    
                else:
                    print(f"Failed to download {drug_name} from {alternate_link}")
                    
print(f"Filtered zipped links and drug names saved to 'filtered_zipped_links_and_names.csv'.")
