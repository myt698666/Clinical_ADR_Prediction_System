import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Load data
file_path = "Important Features Description/Disgenet results/Psych_grouped_by_gene_soc.csv"
df = pd.read_csv(file_path)

# Extract main category from filename (e.g. "Psych" or "Ear")
category_match = re.search(r'/([^/_]+)_grouped_by_gene_soc\.csv$', file_path)
main_category = category_match.group(1) if category_match else None
print(f"Main Category Extracted: {main_category}")

# Load feature importance data
feature_importance_df = pd.read_csv("Feature Importance/feature_importance_Psych.csv")
top_10_genes = feature_importance_df["Feature"].head(10).tolist()  

# Function to get top 2 SOC_ABBREVs + the main category for each gene
def get_top_with_main_category(group):
    # Always include the main category if it exists
    main_row = group[group["SOC_ABBREV"] == main_category]
    other_rows = group[group["SOC_ABBREV"] != main_category]
    top_2_others = other_rows.nlargest(2, "Count")
    return pd.concat([main_row, top_2_others])

# Apply grouping
df_top3 = df.groupby("gene", group_keys=False).apply(get_top_with_main_category)

# Merge to mark top 3 (main + top 2 others)
df = df.merge(df_top3[["gene", "SOC_ABBREV"]], on=["gene", "SOC_ABBREV"], how="left", indicator=True)

# Assign category group labels
df["Rank Group"] = df["SOC_ABBREV"].where(df["_merge"] == "both", "Other")
df.drop(columns=["_merge"], inplace=True)

# Summarise counts
df = df.groupby(["gene", "Rank Group"])["Count"].sum().reset_index()
df = df.sort_values(by=["gene", "Count"], ascending=[True, False])

# Ensure all top genes are included
df["Count"] = df["Count"].astype(float)
all_genes = pd.DataFrame({"gene": top_10_genes})
df = all_genes.merge(df, on="gene", how="left").fillna(0)

# Plotting
unique_groups = df["Rank Group"].unique()
palette = sns.color_palette("Set3", len(unique_groups))
color_map = dict(zip(unique_groups, palette))

# --- Color mapping with brighter green for main category ---
main_color = "#00FF00"  # Bright green (you can adjust the hex code if needed)

# Unique Rank Groups (subcategories)
all_groups = df["Rank Group"].unique().tolist()

# Ensure main category is first
if main_category not in all_groups:
    raise ValueError(f"Main category '{main_category}' not found in Rank Group column.")

# Separate out the other groups, excluding '0'
other_groups = [g for g in all_groups if g != main_category and g != '0']

# Generate a palette for the remaining categories (excluding green and '0')
palette_excluding_main = sns.color_palette("Set3", len(other_groups))

# Combine into a color map: main category is bright green, rest get assigned from palette
color_map = {main_category: main_color}
color_map.update(dict(zip(other_groups, palette_excluding_main)))

# --- Plotting ---
plt.figure(figsize=(12, 6), facecolor='none')
ax = sns.barplot(
    data=df[df["Rank Group"] != '0'],  # Exclude '0' from the data used for plotting
    x="gene",
    y="Count",
    hue="Rank Group",
    palette=color_map
)

# Labels and formatting
plt.title(f"SOC Association Summary for {main_category}", fontsize=28, fontweight='bold')  # Larger and bold title
plt.xlabel("Target", fontsize=24, fontweight='bold')  # Larger and bold x-axis label
plt.ylabel("Count", fontsize=24, fontweight='bold')  # Larger and bold y-axis label
plt.xticks(fontsize=18, fontweight='bold', rotation=45)  # Larger and bold x-ticks
plt.yticks(fontsize=18, fontweight='bold')  # Larger and bold y-ticks

# Legend formatting: Exclude '0' from the legend
handles, labels = ax.get_legend_handles_labels()
filtered_handles_labels = [(h, l) for h, l in zip(handles, labels) if l != '0']
handles, labels = zip(*filtered_handles_labels)
legend = plt.legend(
    handles,
    labels,
    bbox_to_anchor=(0.55, 1),
    loc="upper center",
    ncol=6,
    frameon=False,
    fontsize=16,  # Larger legend font size
    title_fontsize=14  # Larger and bold legend title
)
legend.get_title().set_fontweight('bold')  # Make the legend title bold

plt.subplots_adjust(top=0.75)
plt.show()
