import time
from chembl_webresource_client.new_client import new_client
import pandas as pd

# Initialize ChEMBL client resources
molecule = new_client.molecule
activity = new_client.activity
target = new_client.target

# Example InChIKeys (replace these with your actual keys)
inchi_keys = [
    "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",  # Aspirin
    "RZVAJINKPMORJF-UHFFFAOYSA-N",  # Paracetamol
    "HEFNNWSXXWATRW-UHFFFAOYSA-N",  # Ibuprofen
    "RZVAJINKPMORJF-UHFFFAOYSA-N",  # Acetaminophen
    "RYYVLZVUVIJVGH-UHFFFAOYSA-N",  # Caffeine
]

# Retry mechanism
def fetch_with_retries(fetch_function, *args, max_retries=3, delay=2, **kwargs):
    for attempt in range(max_retries):
        try:
            return fetch_function(*args, **kwargs)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise

# Create a list to store results
results = []

for inchi_key in inchi_keys:
    try:
        # Fetch molecule by InChIKey with retries
        drug_data = fetch_with_retries(
            molecule.filter, molecule_structures__standard_inchi_key=inchi_key
        )
        if not drug_data:
            print(f"No data found for InChIKey: {inchi_key}")
            continue  # Skip if no data found for the InChIKey

        drug_chembl_id = drug_data[0].get('molecule_chembl_id', "Unknown")
        drug_name = drug_data[0].get('pref_name', "Unknown")
        molecular_weight = drug_data[0].get('molecule_properties', {}).get('full_mwt', "Unknown")

        # Fetch bioactivities for the drug with retries
        bioactivities = fetch_with_retries(activity.filter, molecule_chembl_id=drug_chembl_id)

        for entry in bioactivities:
            # Only include relevant data
            if entry.get('standard_value') is not None:
                results.append({
                    'Drug': drug_name,
                    'Target_ID': entry.get('target_chembl_id'),
                    'Metric': entry.get('standard_type'),
                    'Value': float(entry.get('standard_value')),
                    'Molecular_Weight': molecular_weight
                })

    except Exception as e:
        print(f"Error processing InChIKey {inchi_key}: {e}")

# Convert results into a DataFrame
df = pd.DataFrame(results)

if not df.empty:
    # Resolve duplicates by grouping and aggregating
    df = df.groupby(['Drug', 'Target_ID', 'Metric'], as_index=False).agg({
        'Value': 'mean',  # Replace 'mean' with 'max', 'min', or another if needed
        'Molecular_Weight': 'first'
    })

    # Filter for specific metrics (e.g., Ki and Cmax)
    ki_df = df[df['Metric'] == 'Ki']
    cmax_df = df[df['Metric'] == 'Cmax']

    # Merge Ki and Cmax data for each drug
    merged_df = pd.merge(
        ki_df[['Drug', 'Target_ID', 'Value']],
        cmax_df[['Drug', 'Value']],
        on='Drug',
        how='left',
        suffixes=('_Ki', '_Cmax')
    )

    # Add molecular weight
    merged_df = merged_df.merge(
        ki_df[['Drug', 'Molecular_Weight']].drop_duplicates(),
        on='Drug',
        how='left'
    )

    # Replace Target IDs with Target Names
    target_names = {}
    for target_id in merged_df['Target_ID'].unique():
        if pd.notna(target_id):
            try:
                target_data = target.filter(target_chembl_id=target_id)
                target_names[target_id] = target_data[0].get('pref_name', target_id) if target_data else target_id
            except Exception as e:
                print(f"Error fetching target name for {target_id}: {e}")
                target_names[target_id] = target_id

    merged_df['Target'] = merged_df['Target_ID'].map(target_names)

    # Create the pivot table
    pivot_table = merged_df.pivot_table(
        index='Drug',
        columns='Target',
        values='Value_Ki',
        aggfunc='mean'  # Handle duplicates in the pivot by averaging
    )
    pivot_table = pivot_table.fillna("No Data")

    # Display the final table
    print(pivot_table)

    # Save to a CSV file (optional)
    pivot_table.to_csv('drug_target_ki_cmax_table.csv')

else:
    print("No data retrieved. Please check the InChIKeys or API response.")