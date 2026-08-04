import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime

def calculate_ror(group):
    a = group['event_count'].sum()
    b = group['total_drug_reports'].iloc[0] - a
    c = group['total_event_report'].iloc[0] - a
    d = group['total_reports'].iloc[0] - (a + b + c)

    ror = (a / b) / (c / d)
    prr = (a / (a + b)) / (c / (c + d))
    se_log_ror = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_lower = np.exp(np.log(ror) - 1.96 * se_log_ror)
    ci_upper = np.exp(np.log(ror) + 1.96 * se_log_ror)

    return pd.Series({
        'ROR': ror,
        'PRR': prr,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper,
        'event_count': a
    })

def perform_disproportionality_analysis(data):
    results = data.groupby(['Drug', 'SOC_ABBREV']).apply(calculate_ror).reset_index()

    results['Significant'] = results.apply(lambda row: 
        (row['ROR'] > 2) and (row['PRR'] > 2) and (row['CI_lower'] > 1) and (row['event_count'] > 3), axis=1).astype(int)

    return results

def create_significance_matrix(results):
    significance_matrix = results.pivot_table(
        index='Drug',
        columns='SOC_ABBREV',
        values='Significant',
        fill_value=0
    )
    return significance_matrix

def process_for_category(data, filter_column, filter_value):
    filtered_data = data[data[filter_column] == filter_value]

    # Group and calculate metrics
    grouped_data = filtered_data.groupby(['Drug', 'SOC_ABBREV']).size().reset_index(name='event_count')
    total_counts_drug = filtered_data.groupby('Drug').size().reset_index(name='total_drug_reports')
    grouped_data = pd.merge(grouped_data, total_counts_drug, on='Drug', how='left')

    total_counts_soc = grouped_data.groupby('SOC_ABBREV')['event_count'].sum().reset_index(name='total_event_report')
    grouped_data = pd.merge(grouped_data, total_counts_soc, on='SOC_ABBREV', how='left')

    grouped_data['total_reports'] = filtered_data.shape[0]

    # Perform disproportionality analysis
    results = perform_disproportionality_analysis(grouped_data)

    # Create significance matrix
    significance_matrix = create_significance_matrix(results)

    # Save matrix for the category
    output_file = f'ADR_sex_age_analysis/matrix_{filter_column}_{filter_value}.csv'
    significance_matrix.to_csv(output_file)
    print(f"Significance matrix saved to {output_file}")

def main():
    # Load the data
    file_path = 'ADR_Summary/All_ADR_Data.csv'
    data = pd.read_csv(file_path)

    # Filter data to exclude unwanted SOC_ABBREV values
    unwanted_values = ['Genrl', 'Inv', 'Inj&P', 'SocCi', 'Surg']
    data_filtered = data[~data['SOC_ABBREV'].isin(unwanted_values)]

    # Exclude AGE_10 values above 150
    if 'AGE_10' in data_filtered.columns:
        data_filtered = data_filtered[data_filtered['AGE_10'] <= 150]

    # Process for each unique value in SEX
    if 'SEX' in data_filtered.columns:
        unique_sex_values = data_filtered['SEX'].dropna().unique()
        for sex_value in unique_sex_values:
            print(f"Processing for SEX: {sex_value}")
            process_for_category(data_filtered, 'SEX', sex_value)

    # Process for each unique value in AGE_10
    if 'AGE_10' in data_filtered.columns:
        unique_age_values = data_filtered['AGE_10'].dropna().unique()
        for age_value in unique_age_values:
            print(f"Processing for AGE_10: {age_value}")
            process_for_category(data_filtered, 'AGE_10', age_value)

if __name__ == "__main__":
    main()
