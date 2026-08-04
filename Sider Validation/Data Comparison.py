import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib_venn import venn2

# -----------------------------
# Load the datasets
df_control = pd.read_csv("ADR_Summary/SOC_significance_matrix_Sider_Comparison.csv")
df_validation = pd.read_csv("Sider Validation/Significance_Matrix_Lower.csv")
# Drop the 'Prod' column from both DataFrames
df_control = df_control.drop(columns='Prod')
df_validation = df_validation.drop(columns='Prod')

# Rename 'Drug1_Name' in df_validation to match 'Drug' in df_control
df_validation = df_validation.rename(columns={'Drug1_Name': 'Drug'})

# Merge datasets on 'Drug'
df_merged = df_control.merge(df_validation, on='Drug', suffixes=('_control', '_validation'))

# Filter to keep only rows where 'Drug' is in both datasets
df_control = df_control.merge(df_validation[['Drug']], on='Drug', how='inner')
df_validation = df_validation.merge(df_control[['Drug']], on='Drug', how='inner')

# (Optional) Filter to SSRI drugs
ssri_drugs = ['Citalopram','Dapoxetine','Escitalopram','Fluoxetine',
              'Fluvoxamine','Paroxetine','Sertraline','Vortioxetine']
# Uncomment if needed:
# df_control = df_control[df_control['Drug'].isin(ssri_drugs)]
# df_validation = df_validation[df_validation['Drug'].isin(ssri_drugs)]
# df_merged = df_merged[df_merged['Drug'].isin(ssri_drugs)]

# Determine which columns are numeric in the control dataset
numeric_cols = df_control.columns[df_control.dtypes != 'object']

# -----------------------------
# (Optional) Plot the original control and validation heatmaps

# Create heatmaps for control and validation (using only numeric data)
df_control_heatmap = df_control.set_index('Drug')[numeric_cols]
df_validation_heatmap = df_validation.set_index('Drug')[numeric_cols]

#fig, axes = plt.subplots(1, 2, figsize=(18, 8))
#custom_cmap = LinearSegmentedColormap.from_list("custom", ["#F1F2F3", "#12616E"], N=256)

#sns.heatmap(df_control_heatmap, cmap=custom_cmap, fmt=".2f", center=0, ax=axes[0])
#axes[0].set_title("Control Data Heatmap")

#sns.heatmap(df_validation_heatmap, cmap=custom_cmap, fmt=".2f", center=0, ax=axes[1])
#axes[1].set_title("Validation Data Heatmap")
#plt.tight_layout()
#plt.show()

# -----------------------------
# Create a Difference DataFrame with Coded Values
# We will assign a code for each cell based on the following rules:
#   - First, binarize the values (e.g. positive if > 0, negative otherwise).
#   - Then, for each numeric column:
#         If control == validation:
#             Code = 2 if both positive ("In Both")
#             Code = 4 if both negative (blank)
#         Else:
#             Code = 1 if control is positive (i.e. only in SIDER)
#             Code = 3 if control is negative (i.e. only in YCS)

df_diff_codes = pd.DataFrame()
df_diff_codes['Drug'] = df_merged['Drug']

# Loop over each numeric column
for col in numeric_cols:
    codes = []
    for i, row in df_merged.iterrows():
        control_val = row[f"{col}_control"]
        validation_val = row[f"{col}_validation"]
        # Binarize: assume positive if value > 0, negative otherwise.
        control_bin = 1 if control_val > 0 else 0
        validation_bin = 1 if validation_val > 0 else 0
        
        # Assign code based on the comparison:
        if control_bin == validation_bin:
            code = 2 if control_bin == 1 else 4
        else:
            code = 1 if control_bin == 1 else 3
        codes.append(code)
    df_diff_codes[col] = codes

#------------------------------
# Calculate the similarity and count of all the data
# Calculate counts using the full dataset
In_SIDER_full = (df_diff_codes==1).sum().sum()
In_Both_Sig_full = (df_diff_codes==2).sum().sum()
In_YCS_full = (df_diff_codes==3).sum().sum()
In_Both_NonSig_full = (df_diff_codes==4).sum().sum()
Total_instances_full = In_SIDER_full + In_Both_Sig_full + In_YCS_full + In_Both_NonSig_full

# Percentages based on full data
perc_SIDER_full = (In_SIDER_full / Total_instances_full) * 100
perc_Both_Sig_full = (In_Both_Sig_full / Total_instances_full) * 100
perc_YCS_full = (In_YCS_full / Total_instances_full) * 100
perc_Both_NonSig_full = (In_Both_NonSig_full / Total_instances_full) * 100

# Calculate Jaccard Index (as fraction, not %)
jaccard_index_full = In_Both_Sig_full / (In_Both_Sig_full + In_YCS_full + In_SIDER_full)
print(f'Percentage SIDER (all): {perc_SIDER_full:.2f}\nPercentage YCS (all): {perc_YCS_full:.2f}\nPercentage overlap sig (all): {perc_Both_Sig_full:.2f}\nPercentage non-sig (all): {perc_Both_NonSig_full:.2f}\n')
print(f'Jaccard Index: {jaccard_index_full*100:.2f}')
 

# -----------------------------
# Optionally limit the number of rows shown in the heatmap.
# For example, if you want 3 times the number of columns as rows:
num_samples_to_display = 3 * (len(df_diff_codes.columns) - 1)  # subtract 1 for 'Drug'
df_diff_codes_limited = df_diff_codes.iloc[:num_samples_to_display, :].copy()
print(num_samples_to_display)
all_drugs_count = df_merged.shape[0]
# Set the 'Drug' column as the index so it is not treated as numeric data
df_diff_codes_limited = df_diff_codes_limited.set_index('Drug')

In_SIDER = (df_diff_codes_limited==1).sum().sum()
In_Both_Sig = (df_diff_codes_limited==2).sum().sum()
In_YCS = (df_diff_codes_limited==3).sum().sum()
In_Both_NonSig = (df_diff_codes_limited==4).sum().sum()
Total_instances = In_Both_NonSig+In_Both_Sig+In_SIDER+In_YCS

# Calculate percentages
perc_SIDER = (In_SIDER / Total_instances) * 100
perc_Both_Sig = (In_Both_Sig / Total_instances) * 100
perc_YCS = (In_YCS / Total_instances) * 100
perc_Both_NonSig = (In_Both_NonSig / Total_instances) * 100
perc_drugs = (num_samples_to_display/all_drugs_count*100)

# -----------------------------
# Set font colour to poster colour
plt.rcParams["text.color"] = "#404040"
plt.rcParams["axes.labelcolor"] = "#404040"
plt.rcParams["xtick.color"] = "#404040"
plt.rcParams["ytick.color"] = "#404040"
# Define the custom colormap and norm for our 4 codes.
# We need 4 colors and corresponding boundaries.
cmap_diff = ListedColormap(['#E74C3C', '#27AE60', '#F39C12','#8FC2D8' ])
bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
norm = BoundaryNorm(bounds, cmap_diff.N)

# Create a single figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor='none')

# -----------------------------
# Plot the Heatmap (Left Subplot)
ax_diff = sns.heatmap(
    df_diff_codes_limited.astype(float),
    cmap=cmap_diff,
    norm=norm,
    cbar=True,
    xticklabels=False,
    yticklabels=False,
    square=True,
    ax=axes[0]
)
ax_diff.set_title(
    "A) SIDER/YCS Comparison Heatmap",
    fontsize=20,
    fontweight='bold',
    pad=20  # Adjust padding
)
ax_diff.set_xlabel(f"{len(df_diff_codes.columns)-1} SOCs", fontsize=16, fontweight='bold')
ax_diff.set_ylabel(f"{perc_drugs:.0f}% of Drugs", fontsize=16, fontweight='bold')

# Adjust the colorbar ticks and labels
cbar = ax_diff.collections[0].colorbar
cbar.set_ticks([1, 2, 3, 4])
cbar.set_ticklabels([
    f'SIDER',
    f'Both\nSig',
    f'YCS',
    f'Both\nNon-Sig'
])
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=16, fontweight='bold')

# -----------------------------
# Plot the Venn Diagram (Right Subplot)
only_SIDER = (perc_SIDER_full / 100) * Total_instances_full  # A and not B
only_YCS = (perc_YCS_full / 100) * Total_instances_full      # B and not A
overlap = (perc_Both_Sig_full / 100) * Total_instances_full  # A and B

# Create the Venn diagram
venn = venn2(
    subsets=(
        only_SIDER,  # Only in SIDER (A and not B)
        only_YCS,    # Only in YCS (B and not A)
        overlap      # Overlap (A and B)
    ),
    set_labels=('SIDER', 'YCS'),
    ax=axes[1]
)

# Customize the Venn diagram colors
venn.get_patch_by_id('10').set_facecolor('#E74C3C')  # Red for only SIDER
venn.get_patch_by_id('01').set_facecolor('#F1C40F')  # Yellow for only YCS
venn.get_patch_by_id('11').set_facecolor('#27AE60')  # Green for overlap

# Customize the Venn diagram labels
if venn.get_label_by_id('10'):  # Only in SIDER
    venn.get_label_by_id('10').set_text(f'{int(only_SIDER)}\n({perc_SIDER_full:.1f}%)')
    venn.get_label_by_id('10').set_fontweight('bold')  # Make the text bold
    venn.get_label_by_id('10').set_fontsize(16)  # Increase font size
if venn.get_label_by_id('01'):  # Only in YCS
    venn.get_label_by_id('01').set_text(f'{int(only_YCS)}\n({perc_YCS_full:.1f}%)')
    venn.get_label_by_id('01').set_fontweight('bold')  # Make the text bold
    venn.get_label_by_id('01').set_fontsize(16)  # Increase font size
if venn.get_label_by_id('11'):  # Overlap
    venn.get_label_by_id('11').set_text(f'{int(overlap)}\n({perc_Both_Sig_full:.1f}%)')
    venn.get_label_by_id('11').set_fontweight('bold')  # Make the text bold
    venn.get_label_by_id('11').set_fontsize(16)  # Increase font size

# Customize the set labels (SIDER and YCS)
for label in venn.set_labels:
    if label:  # Check if the label exists
        label.set_fontweight('bold')  # Make the set labels bold
        label.set_fontsize(16)  # Optionally, increase the font size

# Add "Not A or Not B" value to the background of the Venn diagram
axes[1].text(
    0.5, -0.2,
    f"Both Non-Sig:\n{int(In_Both_NonSig_full)}\n({perc_Both_NonSig_full:.1f}%)",
    fontsize=14,
    fontweight='bold',
    ha='center',
    transform=axes[1].transAxes
)

# Add Jaccard Index annotation to the Venn diagram
axes[1].text(
    0.5, -0.3,
    f"Jaccard Index: {jaccard_index_full:.2f}",
    fontsize=14,
    fontweight='bold',
    ha='center',
    transform=axes[1].transAxes
)

# Set the title for the Venn Diagram (Right Subplot)
axes[1].set_title(
    "B) All Drug Data Venn Diagram",
    fontsize=18,
    fontweight='bold',
    pad=20,  # Adjust padding
    y=1.05   # Match the vertical position of the heatmap title
)

# -----------------------------
# Adjust layout and show the figure
plt.show()
