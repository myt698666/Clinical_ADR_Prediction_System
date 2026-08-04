import pandas as pd
import numpy as np

# Load the data
file_path = "Sider Validation/final_analysis_output.csv"
data = pd.read_csv(file_path)

def calculate_ror(group, event_col, drug_total_col, soc_total_col, total_reports_col):
    a = group[event_col].sum()
    b = group[drug_total_col].iloc[0] - a
    c = group[soc_total_col].iloc[0] - a
    d = group[total_reports_col].iloc[0] - (a + b + c)
    
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

def perform_disproportionality_analysis(data, event_col, drug_total_col, soc_total_col, total_reports_col):
    results = data.groupby(['Drug1_Name', 'SOC_ABBREV']).apply(
        calculate_ror, event_col, drug_total_col, soc_total_col, total_reports_col
    ).reset_index()
    
    # Identify significant signals
    results['signal_detected'] = results.apply(lambda row: 
        (row['ROR'] > 2) and (row['PRR'] > 2), axis=1)
    
    return results

# Perform analysis for lower and upper bound
lower_bound_results = perform_disproportionality_analysis(
    data, 'Lower Bound (Group)', 'Lower Bound (Drug Total)', 'Lower Bound (SOC Total)', 'Total Lower Bound'
)
# Determine significance
lower_bound_results['Significant'] = (
    (lower_bound_results['ROR'] > 2) &
    (lower_bound_results['PRR'] > 2) 
).astype(int)

# Create the matrix
significance_matrix_Lower = lower_bound_results.pivot_table(
    index='Drug1_Name',
    columns='SOC_ABBREV',
    values='Significant',
    fill_value=0
)

upper_bound_results = perform_disproportionality_analysis(
    data, 'Upper Bound (Group)', 'Upper Bound (Drug Total)', 'Upper Bound (SOC Total)', 'Total Upper Bound'
)

# Determine significance
upper_bound_results['Significant'] = (
    (upper_bound_results['ROR'] > 2) &
    (upper_bound_results['PRR'] > 2) 
).astype(int)

# Create the matrix
significance_matrix_Upper = upper_bound_results.pivot_table(
    index='Drug1_Name',
    columns='SOC_ABBREV',
    values='Significant',
    fill_value=0
)

# Save results
lower_bound_results.to_csv("Sider Validation/lower_bound_analysis.csv", index=False)
upper_bound_results.to_csv("Sider Validation/upper_bound_analysis.csv", index=False)

significance_matrix_Lower.to_csv("Sider Validation/Significance_Matrix_Lower.csv")
significance_matrix_Upper.to_csv("Sider Validation/Significance_Matrix_Upper.csv")

print("Analysis completed and saved.")
