import pandas as pd
import requests
import time
import os

# Define the base URL for UniProt's REST API
base_url = "https://rest.uniprot.org/uniprotkb/search"

# Load the CSV file
file_name = "STITCH_Identifiers/filtered_protein_chemical_interactions.csv"  # Replace with your new CSV file path
column_name = "Protein Name"  # The column containing protein names

# Read the proteins from the CSV file
df = pd.read_csv(file_name)
proteins = df[column_name].dropna().unique()  # Remove NaN and ensure unique values

# List to store results
results = []

# Load existing results if the temporary file exists
temp_file = "UniProt Protein Data/Temp_Protein_Results.csv"
if os.path.exists(temp_file):
    results_df = pd.read_csv(temp_file)
    results = results_df.to_dict(orient='records')

# Iterate through each protein, append '_HUMAN', and query UniProt
for protein in proteins:
    modified_protein = f"{protein}"
    print(f"Querying for protein: {modified_protein}")
    params = {
        "query": f"{modified_protein} AND (organism_id:9606)",  # Human proteins
        "fields": "protein_name,gene_synonym,cc_function,ft_act_site,cc_pathway,go_f,cc_disease",  # Include comments in fields
        "format": "tsv"
    }
    
    # Make the GET request to the UniProt API
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        if data.get("results"):
            for result in data["results"]:
                protein_name = result.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "N/A")
                chembl_references = result.get("uniProtKBCrossReferences", [])
                chembl_id = next((ref["id"] for ref in chembl_references if ref["database"] == "ChEMBL"), "N/A")
                
                # Extract protein family from comments (SIMILARITY comment type)
                protein_family = "N/A"
                comments = result.get("comments", [])
                for comment in comments:
                    if comment.get("commentType") == "SIMILARITY":
                        for text in comment.get("texts", []):
                            if "family" in text.get("value", "").lower():
                                protein_family = text["value"]
                                break
                
                if chembl_id != "N/A":  # Add to results only if ChEMBL ID is valid
                    results.append({
                        "Protein": protein,
                        "Protein Name": protein_name,
                        "Protein Family": protein_family,  # Added protein family as the 3rd column
                        "ChEMBL ID": chembl_id
                    })
        else:
            print(f"No results found for {protein}")
    else:
        print(f"Error querying {protein}: HTTP {response.status_code}")
    
    # Add a delay of 0.1 seconds between requests
    time.sleep(0.1)
    
    # Append results to the temporary CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(temp_file, index=False)
    print(f"Results appended to {temp_file}")

# Final save of results to a CSV file
output_file = "UniProt Protein Data/Final_Protein_Results.csv"  # Replace with your desired output file path
results_df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
