import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the two model metrics CSV files
df_1 = pd.read_csv('Random Forests/Results/Decision_Tree_metrics.csv')
df_2 = pd.read_csv('Random Forests/Results/SOC_model_metrics.csv')

# Drop the last two columns (y_test and y_pred) from the second file
df_2 = df_2.drop(columns=['y_test', 'y_pred'])

# Add a column to distinguish between the two models
df_1['Model'] = 'Decision Tree'
df_2['Model'] = 'Random Forests'

# Combine the two dataframes into one
combined_df = pd.concat([df_1, df_2])

# Create a long-form DataFrame for the boxplot (melting it for easy plotting)
metrics = ['ROC AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC']
long_df = combined_df.melt(id_vars=['Target', 'Model'], value_vars=metrics, var_name='Metric', value_name='Score')

# Create the boxplot
plt.figure(figsize=(16, 8), facecolor='none')
ax = sns.boxplot(x='Metric', y='Score', hue='Model', data=long_df, showfliers=True, width=0.6, linewidth=1.2,
                 palette=['#1E84B0', '#FF7F0E'], dodge=True)  # Different colors for each model
ax.set_facecolor("#F1F2F3")

# Add stripplot for individual points
sns.stripplot(x='Metric', y='Score', hue='Model', data=long_df, color='black', alpha=0.65, jitter=True, dodge=True)

# Labeling the axes and title
plt.xlabel("Performance Metrics", fontsize=24, fontweight='bold')
plt.ylabel("Score", fontsize=24, fontweight='bold')
plt.title("Comparison of Model Metrics", fontsize=32, fontweight='bold')

# Increase tick label font size
ax.tick_params(axis='both', which='major', labelsize=24, labelrotation=90)

# Adjust legend font size
handles, labels = ax.get_legend_handles_labels()
legend = plt.legend(handles=handles[:2], labels=labels[:2], fontsize=20, title="Model", title_fontsize=22, loc='upper right')
legend.get_title().set_fontweight('bold')  # Make the legend title bold

# Show the plot
plt.show()