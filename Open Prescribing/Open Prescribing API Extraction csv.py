import requests
import csv
import os
import time

def get_total_items_as_csv(drug_name, drug_code, original_drug_name, output_file="Open Prescribing/bnf/all_chemical_prescriptions.csv"):
    # Base URL for the Open Prescribing API
    base_url = "https://openprescribing.net/api/1.0/"
    endpoint = "spending/"

    # API query parameters
    params = {
        "code": drug_code.strip(),  # Drug BNF code (remove any leading/trailing spaces)
        "format": "json"
    }

    try:
        # Send the GET request to the API
        response = requests.get(base_url + endpoint, params=params)
        response.raise_for_status()  # Raise an error for HTTP issues

        # Parse the JSON response
        data = response.json()

        if isinstance(data, list):
            mode = "w" if not os.path.exists(output_file) else "a"  # Write mode for the first run, append for subsequent runs

            with open(output_file, mode=mode, newline="", encoding="utf-8") as file:
                writer = csv.writer(file)

                # Write header row only if the file is empty
                header = ["drug_name", "original_drug", "drug_id", "date", "items", "quantity", "actual_cost"]
                if mode == "w":  # Write header if the mode is 'w'
                    writer.writerow(header)

                # Write data rows
                for entry in data:
                    writer.writerow([  
                        drug_name,                  # Processed drug name from combined_data
                        original_drug_name,         # Original drug name from the 'drug' column
                        drug_code,                  # Use the drug_code as the drug_id
                        entry.get("date", ""),
                        entry.get("items", 0),
                        entry.get("quantity", 0.0),
                        entry.get("actual_cost", 0.0)
                    ])

            print(f"Data successfully saved for drug: {drug_name} (BNF code: {drug_code})")
        else:
            print(f"Unexpected response format for drug code: {drug_code}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return False  # Indicate a failed request

    return True  # Indicate a successful request

# Function to read the grouped BNF codes and their matched drugs
def read_grouped_bnf_codes(file_path):
    grouped_bnf_data = []
    with open(file_path, 'r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            bnf_codes = eval(row['BNF Code'])  # Parse the list from the CSV
            drug_name = row['Matched Drug']   # Matched Drug
            for bnf_code in bnf_codes:
                grouped_bnf_data.append((drug_name, bnf_code))  # Add tuples of drug name and BNF code
    return grouped_bnf_data

# Main execution
if __name__ == "__main__":
    grouped_file_path = "Open Prescribing/bnf/grouped_bnf_codes.csv"  # Path to your grouped BNF codes file

    # Read the grouped BNF codes and matched drugs
    grouped_bnf_data = read_grouped_bnf_codes(grouped_file_path)

    # Delete the file at the start if it exists
    output_file = "Open Prescribing Data/drug_data.csv"
    if os.path.exists(output_file):
        os.remove(output_file)

    total_drugs = len(grouped_bnf_data)  # Get the total number of drug-BNF code pairs
    for idx, (drug_name, drug_code) in enumerate(grouped_bnf_data):
        original_drug_name = drug_name  # Use the matched drug as the original drug name for this case

        success = False
        while not success:  # Keep trying until the request succeeds
            success = get_total_items_as_csv(drug_name, drug_code, original_drug_name, output_file)

            if not success:  # If the request fails, wait for 5 seconds and retry
                print("Request failed. Retrying in 5 seconds...")
                time.sleep(5)  # Sleep for 5 seconds

        # Display progress
        print(f"Progress: {idx + 1}/{total_drugs} drugs processed.")
        time.sleep(0.2)

    print("Finished processing all BNF codes.")
