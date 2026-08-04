import requests
import csv
import time
import pandas as pd

def read_pubchem_cids_from_csv(filename):
    """Read PubChem CIDs from a CSV file."""
    pubchem_ids = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cid = row.get('PubChem CID', '').strip()
            if cid and cid != 'No PubChem CID found':
                pubchem_ids.append(cid)
    return pubchem_ids

def fetch_pubchem_data(cid, retries=3, delay=5):
    """Fetch JSON data from PubChem for a given CID with retries."""
    print(f"Fetching CID: {cid}")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/JSON"
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as e:
            print(f"Error fetching data for CID {cid}: {e}")
            attempt += 1
            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                return {"CID": cid, "Error": str(e)}

import pandas as pd

def extract_data(data, output_csv_path="Drug Information/temporary_output.csv"):
    """Extract required values from the PubChem JSON response."""
    extracted_data = {}
    if 'PC_Compounds' in data:
        compound_data = data['PC_Compounds'][0]
        
        # Extract basic fields from the top-level keys and count object.
        extracted_data['cid'] = compound_data.get('id', {}).get('id', {}).get('cid', 'N/A')
        extracted_data['charge'] = compound_data.get('charge', 'N/A')
        extracted_data['atom_chiral_def'] = compound_data.get('count', {}).get('atom_chiral_def', 'N/A')
        extracted_data['atom_chiral_undef'] = compound_data.get('count', {}).get('atom_chiral_undef', 'N/A')
        extracted_data['bond_chiral_def'] = compound_data.get('count', {}).get('bond_chiral_def', 'N/A')
        extracted_data['bond_chiral_undef'] = compound_data.get('count', {}).get('bond_chiral_undef', 'N/A')
        
        # Extract additional count-based properties.
        extracted_data["Heavy_Atom"] = compound_data.get('count', {}).get('heavy_atom', 'N/A')
        extracted_data["Isotope_Atom"] = compound_data.get('count', {}).get('isotope_atom', 'N/A')
        extracted_data["Covalent_Unit"] = compound_data.get('count', {}).get('covalent_unit', 'N/A')
        
        # Loop through the properties list and extract values.
        props = compound_data.get('props', [])
        for prop in props:
            urn = prop.get('urn', {})
            value = prop.get('value', {})
            label = urn.get('label', 'N/A')
            name = urn.get('name', 'N/A')
            header = f"{label}_{name}".strip('_')
            
            # Rename keys to match the required headers.
            if label == "Log P" and name in ("XLogP3", "XLogP3-AA"):
                header = "Log P_Xlogp3"
            elif header == "Weight_MonoIsotopic":
                header = "Weight_Monoisotopic"
            
            # Get the value from the property.
            if 'ival' in value:
                extracted_data[header] = value['ival']
            elif 'fval' in value:
                extracted_data[header] = value['fval']
            elif 'sval' in value:
                extracted_data[header] = value['sval']
            elif 'binary' in value:
                extracted_data[header] = value['binary']
            else:
                extracted_data[header] = 'N/A'
    
    # Define the required headers for the final output.
    required_headers = {
        "cid", "charge", "atom_chiral_def", "atom_chiral_undef", "bond_chiral_def", "bond_chiral_undef",
        "Heavy_Atom", "Log P_Xlogp3", "Count_Rotatable Bond", "Mass_Exact",
        "Count_Hydrogen Bond Donor", "Topological_Polar Surface Area",
        "Weight_Monoisotopic", "Count_Hydrogen Bond Acceptor", "Covalent_Unit",
        "Compound Complexity_N/A", "Isotope_Atom", "Molecular Weight_N/A"
    }
    
    # Filter the extracted data to include only the required headers.
    filtered_data = {key: extracted_data.get(key, 'N/A') for key in required_headers}
    
    # Convert the extracted data to a pandas DataFrame
    df = pd.DataFrame([filtered_data])
    
    # Write the DataFrame to a CSV file to see the results
    df.to_csv(output_csv_path, index=False, mode='a', header=not pd.io.common.file_exists(output_csv_path))  # Append if the file exists
    
    return df


def write_to_csv(data, filename="Drug Information/pubchem_data.csv"):
    """Write the list of dictionaries to a CSV file."""
    headers = set()
    for row in data:
        headers.update(row.keys())
    headers = list(headers)
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

# --- Main execution ---

# Uncomment the following lines if you wish to read multiple CIDs from a CSV file.
pubchem_ids = read_pubchem_cids_from_csv('Drugs and keys/Drug Master Keys.csv')
all_data = [extract_data(fetch_pubchem_data(cid)) for cid in pubchem_ids]
all_data = pd.DataFrame(all_data)
columns_order = ['cid'] + [col for col in all_data.columns if col != 'cid']
all_data = all_data[columns_order]
write_to_csv(all_data)
print("Data has been written to pubchem_data.csv")

# Test the functions with a single CID.
#cid_to_test = 71771  # Test CID
#data = fetch_pubchem_data(cid_to_test)
#extracted_data = extract_data(data)
#print(extracted_data)
# Convert the extracted data dictionary to a DataFrame
#df = pd.DataFrame([extracted_data])
# Reorder columns to place 'cid' at the front
#columns_order = ['cid'] + [col for col in df.columns if col != 'cid']
#df = df[columns_order]

# Print extracted data (for debugging purposes)
#print("\nExtracted Data for CID 71771:")
#for key, value in extracted_data.items():
    #print(f"{key}: {value}")

# Save the DataFrame to a CSV file
#df.to_csv("Drug Information/pubchem_data1.csv", index=False)
