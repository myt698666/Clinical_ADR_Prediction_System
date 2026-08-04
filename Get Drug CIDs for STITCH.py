import requests
import csv

def fetch_cid(drug_name):
    """
    Fetch the CID for a given drug name using PubChem PUG-REST API.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    endpoint = f"{base_url}/compound/name/{drug_name}/cids/JSON"
    
    try:
        response = requests.get(endpoint)
        response.raise_for_status()  # Raise HTTPError for bad responses
        data = response.json()
        if 'IdentifierList' in data:
            return data['IdentifierList']['CID'][0]  # Return the first CID
        else:
            print(f"No CID found for {drug_name}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching CID for {drug_name}: {e}")
        return None

def get_cids_for_drugs(drugs):
    """
    Retrieve CIDs for a list of drugs.
    """
    drug_cid_mapping = {}
    for drug in drugs:
        print(f"Fetching CID for {drug}...")
        cid = fetch_cid(drug)
        if cid is not None:
            drug_cid_mapping[drug] = f"CIDm{str(cid).zfill(8)}"  # Format CID to 8 digits with CIDm prefix
        else:
            drug_cid_mapping[drug] = None
    return drug_cid_mapping

def save_to_csv(drug_cid_mapping, filename):
    """
    Save drug-CID mapping to a CSV file.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Drug Name", "CID"])  # Header row
        for drug, cid in drug_cid_mapping.items():
            writer.writerow([drug, cid])
    print(f"Saved data to {filename}")

# Example list of drugs
drugs = ['Citalopram', 'Dapoxetine', 'Escitalopram', 'Fluoxetine', 'Fluvoxamine', 'Paroxetine', 'Sertraline', 'Vortioxetine']

# Fetch CIDs
cids = get_cids_for_drugs(drugs)

# Save to CSV
save_to_csv(cids, "Drug_CIDs.csv")