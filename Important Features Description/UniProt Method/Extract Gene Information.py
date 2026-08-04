import os
import glob
import pandas as pd
import requests
import time
from io import StringIO

# Define directories
input_dir = "Feature Importance/"
output_dir = "Important Features Description/Model Descriptions/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the UniProt API base URL
base_url = "https://rest.uniprot.org/uniprotkb/search"

# Get a list of all CSV files in the input directory
csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

# Iterate through each CSV file
for input_file in csv_files:
    # Extract the dataset name from the file name
    dataset_name = os.path.basename(input_file).replace("feature_importance_", "").replace(".csv", "")
    output_file = os.path.join(output_dir, f"{dataset_name}.csv")
    
    print(f"\nProcessing file: {input_file}")
    
    # Load the CSV file containing feature importance
    df = pd.read_csv(input_file)
    
    # Extract the top 10 important features
    top_10_features = df["Feature"].head(10).tolist()
    
    # List to store DataFrames from each query
    results_dfs = []
    
    # Iterate through each feature and query UniProt using TSV output
    for feature in top_10_features:
        print(f"Querying UniProt for: {feature}")
        
        params = {
            "query": f"{feature} AND (organism_id:9606)",  # Human proteins
            "fields": "accession,protein_name,gene_synonym,cc_function,ft_act_site,cc_pathway,go_f,cc_disease,xref_chembl",
            "format": "tsv"
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            tsv_data = response.text
            
            # Parse the TSV data into a DataFrame
            df_tsv = pd.read_csv(StringIO(tsv_data), sep="\t")
            if df_tsv.empty:
                print(f"No results found for {feature}")
            else:
                # Optionally, add a column for the feature (Gene) for later reference
                df_tsv.insert(0, "Gene", feature)
                results_dfs.append(df_tsv)
        else:
            print(f"Error querying {feature}: HTTP {response.status_code}")
        
        # Delay to avoid rate limits
        time.sleep(0.1)
    
    # Combine all results into one DataFrame and save to CSV
    if results_dfs:
        combined_df = pd.concat(results_dfs, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print(f"No data was retrieved for {dataset_name}.")
