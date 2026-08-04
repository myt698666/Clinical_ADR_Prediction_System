import pandas as pd
import requests
import json

# Define a function to fetch STRING IDs
def fetch_string_id(protein_name, species_id=9606):
    print(f"Fetching STRING ID for protein: {protein_name}")  # Print statement to track progress
    url = f"https://string-db.org/api/json/get_string_ids?identifiers={protein_name}&species={species_id}"
    response = requests.get(url)
    if response.status_code == 200:
        result = json.loads(response.text)
        if result and 'stringId' in result[0]:
            return result[0]['stringId']
    return None

# Load your data (assuming Protein_Results.csv contains a 'Protein' column)
print("Loading data from Protein_Results.csv...")
df = pd.read_csv('Protein_Results.csv')

# Ensure 'Protein' column exists
if 'Protein' not in df.columns:
    raise ValueError("The DataFrame does not have a 'Protein' column.")

# Remove duplicate protein names
df = df.drop_duplicates(subset=['Protein'])

# Create a new column for STRING IDs
print("Fetching STRING IDs for proteins...")
df['STRING_ID'] = df['Protein'].apply(fetch_string_id)

# Save the updated DataFrame to a new file
print("Saving the updated DataFrame to Protein_Results_with_STRING_IDs.csv...")
df.to_csv('Protein_Results_with_STRING_IDs.csv', index=False)

print("STRING IDs have been added and saved to Protein_Results_with_STRING_IDs.csv")
print("Process complete.")