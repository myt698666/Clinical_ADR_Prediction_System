import spacy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re

def clean_disease(text):
    # Remove curly braces and single quotes
    text = text.replace("{", "").replace("}", "").replace("'", "")
    # Remove any content starting with a '[' (and the '[' itself) until the end of the string
    text = re.sub(r'\[.*$', '', text)
    # Optionally remove extra whitespace
    return text.strip()

# Load spaCy model with word vectors (e.g., 'en_core_web_md')
nlp = spacy.load('en_core_web_md')

# Load the genes file and the PT mapping file
genes_file = 'Important Features Description/Disease Summary/Psych.csv'
pt_file = 'ADR_Summary/ADR_MedDRA_Key_Mapping.csv'

genes_df = pd.read_csv(genes_file)
pt_df = pd.read_csv(pt_file)

# Precompute PT term vectors (assuming the column containing PT terms is named 'PT')
pt_terms = pt_df['PT'].tolist()
pt_vectors = []
for idx, term in enumerate(pt_terms, start=1):
    print(f"Processing PT term {idx}/{len(pt_terms)}: {term}")
    vector = nlp(term).vector
    pt_vectors.append(vector)


# Prepare list to store the matching results
results = []
total_rows = genes_df.shape[0]

# Process each row in the genes file
for idx, row in genes_df.iterrows():
    # Print progress for each gene row
    print(f"Processing row {idx + 1}/{total_rows}: Gene {row['Gene']}")
    gene = row['Gene']
    
    # Split the diseases column into individual disease strings.
    # Adjust the column name if it differs.
    diseases = [clean_disease(d.strip()) for d in row['disease_list'].split(',')]
    
    # Process each disease for the current gene
    for disease in diseases:
        # Print progress for each disease being processed
        print(f"    Processing disease: '{disease}'")
        
        # Compute the vector for the disease string
        disease_vec = nlp(disease).vector.reshape(1, -1)
        
        # Compute cosine similarity with each precomputed PT vector
        similarities = cosine_similarity(disease_vec, pt_vectors)[0]
        
        # Identify the best matching PT term
        best_idx = similarities.argmax()
        best_match = pt_terms[best_idx]
        best_score = similarities[best_idx]
        
        # Append the result for this disease
        results.append({
            'gene': gene,
            'input_disease': disease,
            'PT_match': best_match,
            'score': best_score
        })

# Convert the results list into a DataFrame
results_df = pd.DataFrame(results)

# Save the results to the specified CSV file
output_file = 'Important Features Description/NLP MedDRA Summary/Psych_NLP.csv'
results_df.to_csv(output_file, index=False)

print(f"Processing complete. Output saved to '{output_file}'")
