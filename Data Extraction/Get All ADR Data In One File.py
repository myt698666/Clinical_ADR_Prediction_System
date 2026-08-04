import os
import pandas as pd

# Path to the folder containing all ADR event files
folder_path = "All_ADR_Event_Files"

# List all files in the folder
all_files = os.listdir(folder_path)

# Group files based on the common name prefix
file_groups = {}
for file in all_files:
    if file.endswith('.csv'):
        prefix = file.split('_')[0]
        if prefix not in file_groups:
            file_groups[prefix] = {}
        if "_case.csv" in file:
            file_groups[prefix]["case"] = os.path.join(folder_path, file)
        elif "_drug.csv" in file:
            file_groups[prefix]["drug"] = os.path.join(folder_path, file)
        elif "_event.csv" in file:
            file_groups[prefix]["event"] = os.path.join(folder_path, file)

# Initialize a list to store merged dataframes for all groups
merged_dataframes = []

# Process each group
for idx, (prefix, files) in enumerate(file_groups.items(), start=1):
    print(f"Processing Drug {idx}: {prefix}")
    
    # Load each file
    case_file = pd.read_csv(files["case"])
    drug_file = pd.read_csv(files["drug"])
    event_file = pd.read_csv(files["event"])
    
    # Step 1: Remove duplicate rows in the 'ADR' column in _drug.csv
    drug_file = drug_file.drop_duplicates(subset=['ADR'])
    
    # Step 2: Merge _case.csv and _drug.csv on 'ADR' column
    case_drug_merged = pd.merge(case_file, drug_file, on="ADR", how="inner")
    
    # Step 3: Merge the result with _event.csv on 'ADR' column using a left join
    final_merged = pd.merge(case_drug_merged, event_file, on="ADR", how="left")
    
    # Step 4: Add a 'Drug' column with the prefix as its value
    final_merged.insert(0, 'Drug', prefix)
    
    # Append to the list of merged dataframes
    merged_dataframes.append(final_merged)

# Combine all merged dataframes into one for final visualization
final_data = pd.concat(merged_dataframes, ignore_index=True)

output_folder = "ADR_Summary"

# Save the final data to a CSV file
output_path = os.path.join(output_folder, "All_ADR_Data.csv")
final_data.to_csv(output_path, index=False)

# Confirmation message
print(f"All merged data saved successfully to {output_path}")
