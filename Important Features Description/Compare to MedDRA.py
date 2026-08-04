import pandas as pd
from fuzzywuzzy import process

# Load ADR MedDRA Key Mapping CSV
adr_mapping = pd.read_csv(
    "ADR_Summary/ADR_MedDRA_Key_Mapping.csv",
    dtype=str
)

# Load Gene Disease CSV
gene_disease = pd.read_csv(
    "Important Features Description/Disgenet results/gene_disease_Psych.csv",
    dtype={"assocID": "Int64", "geneNcbiID": "Int64", "geneDSI": float, "geneDPI": float, "score": float, 
           "yearInitial": "Int64", "yearFinal": "Int64", "diseaseType": str}
)

# Function for fuzzy matching
def fuzzy_merge(left_df, right_df, left_on, right_on, threshold=45):
    matches = []
    for item in left_df[left_on].dropna().unique():
        best_match, score = process.extractOne(item, right_df[right_on].dropna().unique())
        print(f"{item}, {best_match}")
        if score >= threshold:
            matches.append((item, best_match))

    match_dict = dict(matches)
    left_df["Matched"] = left_df[left_on].map(match_dict)
    return left_df.merge(right_df, left_on="Matched", right_on=right_on, how="left").drop(columns=["Matched"])

# Perform fuzzy join on 'disease' and 'PT'
merged_df = fuzzy_merge(gene_disease, adr_mapping, left_on="disease", right_on="PT", threshold=45)

# Reorder columns
column_order = ["gene", "disease", "PT", "assocID", "geneNcbiID", "geneDSI", "geneDPI", "score", 
                "yearInitial", "yearFinal", "diseaseType", "HLT", "HLGT", "SOC_ABBREV"]
merged_df = merged_df[column_order]

# Group by gene and SOC_ABBREV, then count rows
grouped_by_gene_soc = merged_df.groupby(["gene", "SOC_ABBREV"]).size().reset_index(name="Count")

# Group by SOC_ABBREV only
grouped_by_soc = merged_df.groupby(["SOC_ABBREV"]).size().reset_index(name="Count")

# Save outputs
merged_df.to_csv("Important Features Description/Disease Summary/Psych_merged_output.csv", index=False)
grouped_by_gene_soc.to_csv("Important Features Description/Disgenet results/Psych_grouped_by_gene_soc.csv", index=False)
grouped_by_soc.to_csv("Important Features Description/Disgenet results/Psych_grouped_by_soc.csv", index=False)

print("Processing complete. Files saved.")
