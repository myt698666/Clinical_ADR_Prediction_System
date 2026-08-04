import pandas as pd

# Load the CSV file
file_path = "STITCH_Identifiers/filtered_protein_chemical_interactions.csv"
df = pd.read_csv(file_path)

# Count occurrences of each protein
protein_counts = df['Protein Name'].value_counts()

# Get the top 500 proteins by count
top_500_proteins = protein_counts.head(500).index

# Filter the original DataFrame to include only rows with the top 500 proteins
filtered_df = df[df['Protein Name'].isin(top_500_proteins)]

# Calculate the total count of interactions for the top 500 proteins
total_interactions = len(filtered_df)

# Create the pivot table
filtered_matrix = filtered_df.pivot_table(
    index='Matched Drug', 
    columns='Protein Name', 
    values='Confidence', 
    fill_value=0
)

# Save the resulting matrix to a new CSV file
output_path = "STITCH_Identifiers/top_500_protein_drug_interaction_matrix.csv"
filtered_matrix.to_csv(output_path)

# Print the total interaction count
print(f"The filtered matrix has been saved to {output_path}")
print(f"The total number of interactions for the top 500 proteins is: {total_interactions}")
