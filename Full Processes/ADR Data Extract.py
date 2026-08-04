import time
import csv
import subprocess

# List of Python files to execute
python_files = ["Data Extraction/All Drugs ADR Data.py", "Data Extraction/Get All ADR Data In One File.py", "Open Prescribing/Filter BNF Snowmed mapping file.py", "Open Prescribing/Get BNF Codes grouped.py", "Open Prescribing/Open Prescribing API Extraction csv.py","Open Prescribing Data/Get Prescribed summary.py"]

# CSV file to store results
output_csv = "Timings/ADR Extraction Execution Times.csv"

# Open the CSV file for writing
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Script Name", "Execution Time (seconds)"])

    # Loop through each Python file
    for py_file in python_files:
        start_time = time.time()  # Start timing
        try:
            # Run the Python file
            subprocess.run(["python", py_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {py_file}: {e}")
        
        end_time = time.time()  # End timing

        # Calculate execution time
        execution_time = end_time - start_time

        # Write to CSV
        writer.writerow([py_file, execution_time])
        print(f"{py_file} executed in {execution_time:.2f} seconds")
