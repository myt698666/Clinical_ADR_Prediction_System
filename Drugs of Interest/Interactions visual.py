import pandas as pd
import os
import plotly.graph_objects as go

# Load the data
matching_signals_dir = "Drugs of Interest/Target associations/Matching Signals"
edges = []

for file_name in os.listdir(matching_signals_dir):
    if file_name.endswith(".csv"):
        target = file_name.replace("_matching signals.csv", "")
        df = pd.read_csv(os.path.join(matching_signals_dir, file_name))
        drugs = df.iloc[:, 0].dropna().unique()
        for drug in drugs:
            edges.append((target, drug))

# Create list of unique nodes
targets = list({e[0] for e in edges})
drugs = list({e[1] for e in edges})
all_nodes = targets + drugs

# Mapping to indices
node_indices = {name: i for i, name in enumerate(all_nodes)}
source = [node_indices[src] for src, dst in edges]
target = [node_indices[dst] for src, dst in edges]

# Generate labels and colors
colors = ['red'] * len(targets) + ['blue'] * len(drugs)

# Create Sankey diagram
fig = go.Figure(data=[go.Sankey(
    arrangement='snap',
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_nodes,
        color=colors
    ),
    link=dict(
        source=source,
        target=target,
        value=[1] * len(edges)
    ))])


fig.update_layout(title_text="Chord-style Drug–Target Links", font_size=12)
fig.show()
