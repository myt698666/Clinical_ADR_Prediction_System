import pandas as pd
from datetime import datetime

# Load the CSV file
file_path = 'ADR_Summary/All_ADR_Data.csv'
data = pd.read_csv(file_path)

# Get the current year
current_year = datetime.now().year

# Filter the data to only include events from the last 5 years
# data = data[data['RECVD_YEAR'] >= current_year - 5]

# Filter the data to exclude unwanted SOC_ABBREV values
unwanted_values = ['Genrl', 'Inv', 'Inj&P', 'SocCi', 'Surg']
data_filtered = data[~data['SOC_ABBREV'].isin(unwanted_values)]

# Group by 'Drug' and 'SOC_ABBREV' and count the occurrences
grouped_data = data_filtered.groupby(['Drug', 'SOC_ABBREV']).size().reset_index(name='event_count')

# Group by 'Drug' and count occurrences
total_counts_drug = data_filtered.groupby('Drug').size().reset_index(name='total_drug_reports')
grouped_data = pd.merge(grouped_data, total_counts_drug, on='Drug', how='left')

# Merge to add the total counts for each 'SOC_ABBREV'
total_counts_soc = grouped_data.groupby('SOC_ABBREV')['event_count'].sum().reset_index(name='total_event_report')
grouped_data = pd.merge(grouped_data, total_counts_soc, on='SOC_ABBREV', how='left')

# Add a column for the total of all reports
grouped_data['total_reports'] = data_filtered.shape[0]  # Total number of rows after filtering

# Optionally, save the result to a new CSV file
grouped_data.to_csv('ADR_Summary/Grouped_Drug_SOC.csv', index=False)

# Print the grouped data with totals
print(grouped_data)
