import requests
import json
import time
import csv
import urllib3
import pandas as pd

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Your API key for DisGeNET
API_KEY = "3098ae1f-d73f-40ea-80e7-2f521c10d2a4"

# Read the CSV file from the 'Feature Importance' folder
# The CSV file is expected to have a column named 'Feature' containing gene symbols
feature_file = "Feature Importance/feature_importance_Psych.csv"
df = pd.read_csv(feature_file)

# Extract the top 10 genes from the 'Feature' column
gene_symbols = df["Feature"].head(10).tolist()
print("Genes to query:", gene_symbols)

# DisGeNET endpoint for GDA summary
endpoint = "https://api.disgenet.com/api/v1/gda/summary"

# Name of the CSV file where the results will be saved
output_file = "Important Features Description/Disgenet results/gene_disease_Psych.csv"

# Define the CSV columns (feel free to add or remove fields as desired)
fieldnames = [
    "gene",
    "disease",
    "assocID",
    "geneNcbiID",
    "geneDSI",
    "geneDPI",
    "score",
    "yearInitial",
    "yearFinal",
    "diseaseType"
]

# Open the CSV file for writing
with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
    # Create a DictWriter using the defined fieldnames.
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the header only once.
    writer.writeheader()
    
    # Iterate over each gene symbol
    for gene in gene_symbols:
        print(f"\nQuerying gene: {gene}")
        # Set up query parameters
        params = {
            "gene_symbol": gene,
            "page_number": "0",  # retrieving the first page
            "source": "CURATED"
        }
        
        # Set up HTTP headers
        headers = {
            "Authorization": API_KEY,
            "accept": "application/json"
        }
        
        #sleep for 2 seconds to not overload API
        time.sleep(2)

        # Make the API request
        response = requests.get(endpoint, params=params, headers=headers, verify=True)
        print("Status code:", response.status_code)
        
        # Handle rate limit (HTTP 429)
        if not response.ok and response.status_code == 429:
            while not response.ok:
                wait_time = int(response.headers.get('x-rate-limit-retry-after-seconds', 1))
                print(f"Rate limit reached. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                print("Retrying query...")
                response = requests.get(endpoint, params=params, headers=headers, verify=False)
                if response.ok:
                    break
        
        # If the response is successful, process it
        if response.ok:
            data = response.json()
            if data.get("status") == "OK" and "payload" in data:
                for entry in data["payload"]:
                    # Build a row with the fields you want to extract.
                    row = {
                        "gene": gene,
                        "disease": entry.get("diseaseName", "N/A"),
                        "assocID": entry.get("assocID", "N/A"),
                        "geneNcbiID": entry.get("geneNcbiID", "N/A"),
                        "geneDSI": entry.get("geneDSI", "N/A"),
                        "geneDPI": entry.get("geneDPI", "N/A"),
                        "score": entry.get("score", "N/A"),
                        "yearInitial": entry.get("yearInitial", "N/A"),
                        "yearFinal": entry.get("yearFinal", "N/A"),
                        "diseaseType": entry.get("diseaseType", "N/A")
                    }
                    writer.writerow(row)
                    print(f"Wrote row: {row}")
            else:
                print(f"No valid data found for gene: {gene}")
        else:
            print(f"Failed to retrieve data for gene: {gene} (HTTP {response.status_code})")

print(f"CSV file '{output_file}' created successfully!")
