import pandas as pd
import requests
import time

# Define the base URL for UniProt's REST API
base_url = "https://rest.uniprot.org/uniprotkb/search"

# Load the Excel file
file_name = "STITCH_Identifiers/filtered_protein_chemical_interactions.csv"  # Replace with your file path
sheet_name = "STITCH data"  # Replace with your sheet name
column_name = "Protein Name"  # The column containing protein names

# Read the proteins from the CSV file
df = pd.read_csv(file_name)
proteins = df[column_name].dropna().unique()  # Remove NaN and ensure unique values

# List to store results
results = []

# Iterate through each protein and query UniProt
for protein in proteins:
    print(f"Querying for protein: {protein}")
    params = {
        "query": protein,
        "fields": "accession,id,protein_name,go_id",  # Specify desired fields
        "format": "json"  # Request JSON response
    }
    
    # Make the GET request to the UniProt API
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Extract and save results
        if data.get("results"):
            for result in data["results"]:
                accession = result.get("primaryAccession", "N/A")
                protein_name = result.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "N/A")
                go_terms = [term["id"] for term in result.get("go", {}).get("biologicalProcess", [])]
                results.append({
                    "Protein": protein,
                    "Accession": accession,
                    "Name": protein_name,
                    "GO Terms": "; ".join(go_terms)
                })
        else:
            results.append({
                "Protein": protein,
                "Accession": "N/A",
                "Name": "No result",
                "GO Terms": "N/A"
            })
    else:
        results.append({
            "Protein": protein,
            "Accession": "Error",
            "Name": f"HTTP {response.status_code}",
            "GO Terms": response.text
        })
    
    print(results)
    # Add a delay of 0.1 seconds between requests
    time.sleep(0.1)

# Save results to a CSV file
output_file = "Protein_Results.csv"  # Replace with your desired output file path
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")

