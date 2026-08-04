import pandas as pd
from scipy.stats import ttest_ind

# Load the CSV files, skipping the first column
file1 = 'Random Forests/Results/Decision_Tree_metrics.csv'
file2 = 'Random Forests/Results/SOC_model_metrics.csv'

df1 = pd.read_csv(file1).iloc[:, 1:7]  # Select columns 2 to 7
df2 = pd.read_csv(file2).iloc[:, 1:7]

# Print shapes of the DataFrames
print(f"Shape of df1: {df1.shape}")
print(f"Shape of df2: {df2.shape}")

# Check that they have the same shape
assert df1.shape == df2.shape, "DataFrames must have the same shape for paired comparison."

# Perform t-tests column by column
results = []
for i, (col1, col2) in enumerate(zip(df1.columns, df2.columns)):
    t_stat, p_val = ttest_ind(df1[col1], df2[col2], equal_var=False)  # Welch's t-test
    results.append({
        "Metric": col1,
        "t-statistic": t_stat,
        "p-value": p_val,
        "Significant at 0.05?": p_val < 0.05
    })

# Display results
results_df = pd.DataFrame(results)

# Save the results DataFrame as a CSV file
results_df.to_csv('Random Forests/Results/t-test_DT_vs_RF.csv', index=False)

# Display results
print(results_df)
