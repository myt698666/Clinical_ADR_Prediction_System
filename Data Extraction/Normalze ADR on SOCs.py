import pandas as pd

# Load the datasets
adr_summary = pd.read_csv('ADR_Summary/Grouped_Drug_SOC.csv')
prescribed_summary = pd.read_csv('Open Prescribing Data/prescribed summary.csv')

# Perform the inner join and exclude the 'drug_name' column from the result
merged_data = pd.merge(
    adr_summary, 
    prescribed_summary[['drug_name', 'items']],  # Only keep 'drug_name' and 'items' for the merge
    how='inner', 
    left_on='Drug', 
    right_on='drug_name'
).drop(columns=['drug_name'])  # Drop 'drug_name' after the merge

# Create the new 'adrs per 100,000' column
merged_data['adrs per 100,000'] = (merged_data['event_count'] / merged_data['items']) * 100000

# Save the resulting file
merged_data.to_csv('ADR_Summary/Normalized_SOC.csv', index=False)
