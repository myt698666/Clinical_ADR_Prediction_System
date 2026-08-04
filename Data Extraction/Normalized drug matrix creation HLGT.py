import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('ADR_Summary/Normalized_HLGT.csv')

# Filter rows where 'items' > 1000
filtered_df = df[df['items'] > 1000]

# Pivot the table
result_matrix = filtered_df.pivot(index='Drug', columns='HLGT', values='adrs per 100,000').fillna(0)

# Print or save the resulting matrix
print(result_matrix)

# Optionally, save to a CSV file
result_matrix.to_csv('ADR_Summary/Filtered_Normalized_matrix_HLGT.csv')
