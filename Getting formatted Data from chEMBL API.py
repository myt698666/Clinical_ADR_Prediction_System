from chembl_webresource_client.new_client import new_client
import pandas as pd

# Initialize ChEMBL client resources
molecule = new_client.molecule
activity = new_client.activity

# Example InChIKeys (replace these with your actual keys)
inchi_keys = [
    "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",  # Aspirin
    "RZVAJINKPMORJF-UHFFFAOYSA-N",  # Paracetamol
    "HEFNNWSXXWATRW-UHFFFAOYSA-N",  # Ibuprofen
    "RZVAJINKPMORJF-UHFFFAOYSA-N",  # Acetaminophen
    "RYYVLZVUVIJVGH-UHFFFAOYSA-N",  # Caffeine
]

# Create a dictionary to hold the results
resultski = []
resultscmax = []

for inchi_key in inchi_keys:
    # Fetch molecule by InChIKey
    drug_data = molecule.filter(molecule_structures__standard_inchi_key=inchi_key)
    if not drug_data:
        continue  # Skip if no data found for the InChIKey
    
    drug_chembl_id = drug_data[0]['molecule_chembl_id']
    drug_name = drug_data[0].get('pref_name', "Unknown")

    # Fetch bioactivities for the drug
    bioactivities = activity.filter(molecule_chembl_id=drug_chembl_id)
    
    for entry in bioactivities:
        # Filter for Ki values
        if entry.get('standard_type') == 'Ki' and entry.get('standard_value') is not None:
            resultski.append({
                'Drug': drug_name,
                'Target': entry.get('target_name'),
                'Ki': float(entry.get('standard_value')),
            })
    
    for entry in bioactivities:
        # Filter for Ki values
        if entry.get('standard_type') == 'Cmax' and entry.get('standard_value') is not None:
            resultscmax.append({
                'Drug': drug_name,
                'Cmax': float(entry.get('standard_value')),
            })

# Convert results into a DataFrame
dfki = pd.DataFrame(resultski)
dfcmax = pd.DataFrame(resultscmax)

df = pd.merge(dfki,dfcmax,on='drug_name',how='outer')

# Remove duplicates, keeping the first occurrence
df = df.drop_duplicates(subset=['Drug', 'Target'])

# Proceed with pivoting as before
pivot_table = df.pivot(index='Drug', columns='Target', values='Ki')

# Replace NaN with a placeholder if needed (e.g., "No Data")
pivot_table = pivot_table.fillna("No Data")

# Display the pivot table
print(pivot_table)

# Save to a CSV file (optional)
pivot_table.to_csv('drug_target_ki_table.csv')
