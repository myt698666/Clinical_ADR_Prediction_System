import pandas as pd

# File path of the filtered output CSV
input_csv = "Open Prescribing/bnf/bnf_codes.csv"

# Load the filtered data
filtered_bnf_data = pd.read_csv(input_csv)

# Keep only the first 11 characters of the 'BNF Code' column
filtered_bnf_data['BNF Code'] = filtered_bnf_data['BNF Code'].str[:11]

# Group by 'Matched Drug' and remove duplicates
grouped_data = (
    filtered_bnf_data.groupby('Matched Drug')['BNF Code']
    .apply(lambda x: x.drop_duplicates().tolist())  # Remove duplicates
    .reset_index()
)

# Save the grouped data to a new CSV file
grouped_data.to_csv("Open Prescribing/bnf/grouped_bnf_codes.csv", index=False)

# Print success message
print("Grouped data saved to 'grouped_bnf_codes.csv'.")
