import pandas as pd
import time
import re

# Start timer
start_time = time.time()

# File paths
excel_file = "BNF Snomed Mapping data 20241216.xlsx"
csv_file = "Drug InChi Keys/All_drug_Inchi_and_smiles.csv"

# Load the data
bnf_data = pd.read_excel(excel_file)
drug_data = pd.read_csv(csv_file)

# Extract the 'Drug' column as a list
drug_list = drug_data['Drug'].dropna().tolist()

# Create a pattern for the contains filter using '|' as the OR operator
pattern = '|'.join(map(re.escape, drug_list))

# Apply the contains filter
filtered_bnf_data = bnf_data[bnf_data['BNF Name'].str.contains(pattern, na=False, case=False)]

# Add a column for the matched drug
def find_matching_drug(bnf_name):
    for drug in drug_list:
        if drug.lower() in str(bnf_name).lower():
            return drug  # Return the first matching drug
    return None

# Apply the matching function to assign the matched drug
filtered_bnf_data['Matched Drug'] = filtered_bnf_data['BNF Name'].apply(find_matching_drug)

# Save the filtered data to a CSV file
filtered_bnf_data.to_csv("Open Prescribing/bnf/bnf_codes.csv", index=False)

# Stop timer
end_time = time.time()

# Print the time taken
print(f"Filtered data saved to 'bnf codes.csv'. Time taken: {end_time - start_time:.2f} seconds.")
