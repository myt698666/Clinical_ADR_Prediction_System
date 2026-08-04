import pandas as pd

# Load the drug names file
drug_names = pd.read_csv("Sider Validation/drug_names.tsv", sep="\t", header=None, names=["ID", "Drug_Name"])

drug_atc = pd.read_csv("Sider Validation/drug_atc.tsv", sep="\t", header=None, names=["ID","ATC_Code"])
# Load the meddra frequency file with specific column names
meddra_freq = pd.read_csv("Sider Validation/meddra_freq.tsv", sep="\t", header=None, names=["Drug1ID", "Drug2ID", "MedDRA_ID", "Placebo", "Percentage", "Lower bound", "Upper Bound", "Type", "ID_2", "PT"])

# Merge drug names with meddra_freq on Drug1ID
meddra_freq = meddra_freq.merge(drug_names, left_on="Drug1ID", right_on="ID", how="left")
meddra_freq.rename(columns={"Drug_Name": "Drug1_Name"}, inplace=True)
meddra_freq.drop(columns=["ID"], inplace=True)

# Merge drug names with meddra_freq on Drug2ID
meddra_freq = meddra_freq.merge(drug_atc, left_on="Drug2ID", right_on="ID", how="left")
meddra_freq.rename(columns={"ATC_Code": "ATC_Code"}, inplace=True)
meddra_freq.drop(columns=["ID"], inplace=True)

# Load the ADR mapping file
adr_mapping = pd.read_csv("ADR_Summary/ADR_MedDRA_Key_Mapping.csv")

# Merge MedDRA terms with ADR mapping file on the PT column
meddra_freq = meddra_freq.merge(adr_mapping, on="PT", how="left")

# Keep only necessary columns
final_columns = ["Drug1ID", "Drug1_Name", "Drug2ID", "ATC_Code", "MedDRA_ID", "Placebo", "Percentage", "Lower bound", "Upper Bound", "Type", "PT", "HLT", "HLGT", "SOC_ABBREV"]
meddra_freq = meddra_freq[final_columns]

# Filter the data to exclude unwanted SOC_ABBREV values
unwanted_values = ['Genrl', 'Inv', 'Inj&P', 'SocCi', 'Surg']
meddra_freq = meddra_freq[~meddra_freq['SOC_ABBREV'].isin(unwanted_values)]

# Perform analysis calculations
# Total sums of lower and upper bounds
total_lower = meddra_freq["Lower bound"].sum()
total_upper = meddra_freq["Upper Bound"].sum()

# Sum for each unique SOC_ABBREV
soc_totals = meddra_freq.groupby("SOC_ABBREV")[["Lower bound", "Upper Bound"]].sum().reset_index()

# Sum for each unique Drug1_Name
drug_totals = meddra_freq.groupby("Drug1_Name")[["Lower bound", "Upper Bound"]].sum().reset_index()

# Group by Drug1_Name and SOC_ABBREV with sum operations
grouped_df = meddra_freq.groupby(["Drug1_Name", "SOC_ABBREV"])[["Lower bound", "Upper Bound"]].sum().reset_index()

# Merge the totals into the grouped data
final_df = grouped_df.merge(soc_totals, on="SOC_ABBREV", suffixes=("", "_SOC"))
final_df = final_df.merge(drug_totals, on="Drug1_Name", suffixes=("", "_Drug"))

# Rename columns for clarity
final_df.rename(columns={"Lower bound": "Lower Bound (Group)", "Upper Bound": "Upper Bound (Group)",
                          "Lower bound_SOC": "Lower Bound (SOC Total)", "Upper Bound_SOC": "Upper Bound (SOC Total)",
                          "Lower bound_Drug": "Lower Bound (Drug Total)", "Upper Bound_Drug": "Upper Bound (Drug Total)"}, inplace=True)

# Add total lower and upper bounds as new columns
final_df["Total Lower Bound"] = total_lower
final_df["Total Upper Bound"] = total_upper

# Capitalise the first letter of each word in the Drug1_Name column
final_df["Drug1_Name"] = final_df["Drug1_Name"].str.title()

# Reorder the columns so all "Lower Bound" columns are together followed by all "Upper Bound" columns
ordered_columns = [
    "Drug1_Name", "SOC_ABBREV",
    "Lower Bound (Group)", "Lower Bound (SOC Total)", "Lower Bound (Drug Total)", "Total Lower Bound", 
    "Upper Bound (Group)", "Upper Bound (SOC Total)", "Upper Bound (Drug Total)", "Total Upper Bound"
]

# Apply the new column order
final_df = final_df[ordered_columns]
final_df = final_df.sort_values(by='Drug1_Name')

# Save the final merged file
final_df.to_csv("Sider Validation/final_analysis_output.csv", index=False)
