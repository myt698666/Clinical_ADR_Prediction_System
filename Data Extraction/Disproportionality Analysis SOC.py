import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_ror(group):
    a = group['event_count'].sum()
    b = group['total_drug_reports'].iloc[0] - a
    c = group['total_event_report'].iloc[0] - a
    d = group['total_reports'].iloc[0] - (a + b + c)
    
    ror = (a / b) / (c / d)
    prr = (a/(a+b))/(c/(c+d))
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
    
    # Apply conditions with if statements
    results['signal_detected'] = results.apply(lambda row: 
        (row['ROR'] > 2) and (row['CI_lower'] > 1) and (row['event_count'] >= 3), axis=1)
    
    return results

# Load your data
data = pd.read_csv('ADR_Summary/Grouped_Drug_SOC.csv')

# Perform analysis for SOC
soc_results = perform_disproportionality_analysis(data)

# Display the first few rows of the results
print(soc_results.head())

# Summarize the results
summary = soc_results.groupby('Drug')['signal_detected'].sum().sort_values(ascending=False)
print("\nTop 10 drugs with the most signals detected:")
print(summary.head(10))

# Save the results to a CSV file
soc_results.to_csv('ADR_Summary/soc_disproportionality_results.csv', index=False)