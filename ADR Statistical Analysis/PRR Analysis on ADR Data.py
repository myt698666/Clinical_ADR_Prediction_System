import pandas as pd

# Load the CSV file
file_path = 'ADR Events.csv'  # Update with the correct path if needed
data = pd.read_csv(file_path)

# Set threshold for PRR
threshold = 1.0

# Calculate total number of reports for each Drug-SOC pair
drug_soc_counts = data.groupby(['Drug', 'SOC_ABBREV']).size().reset_index(name='Count')

# Calculate total number of reports for each Drug
drug_totals = data['Drug'].value_counts().reset_index()
drug_totals.columns = ['Drug', 'Total_Reports']

# Merge total reports with drug-SOC counts
drug_soc_counts = drug_soc_counts.merge(drug_totals, on='Drug')

# Calculate the proportion of ADRs for each Drug-SOC pair
drug_soc_counts['Proportion_Drug_SOC'] = drug_soc_counts['Count'] / drug_soc_counts['Total_Reports']

# Calculate total number of reports for each SOC across all drugs
soc_totals = data['SOC_ABBREV'].value_counts().reset_index()
soc_totals.columns = ['SOC_ABBREV', 'Total_SOC_Reports']

# Merge SOC totals with drug-SOC counts
drug_soc_counts = drug_soc_counts.merge(soc_totals, on='SOC_ABBREV')

# Calculate the proportion of ADRs for each SOC across all drugs
total_reports = len(data)
drug_soc_counts['Proportion_SOC'] = drug_soc_counts['Total_SOC_Reports'] / total_reports

# Calculate PRR
drug_soc_counts['PRR'] = drug_soc_counts['Proportion_Drug_SOC'] / drug_soc_counts['Proportion_SOC']

# Apply threshold to determine significant PRR values
drug_soc_counts['Significant'] = (drug_soc_counts['PRR'] > threshold).astype(int)

# Pivot the table to have Drugs as rows and SOC_ABBREV as columns
result_table = drug_soc_counts.pivot(index='Drug', columns='SOC_ABBREV', values='Significant').fillna(0).astype(int)

# Save the final table to a CSV file
result_table.to_csv('ADR Statistical Analysis/PRR_Statistical Analysis_on_ADRs_Result.csv')

print("Analysis complete. Results saved to 'PRR_Analysis_Result.csv'.")