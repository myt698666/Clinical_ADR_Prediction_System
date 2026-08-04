import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("Initializing clinical toxicity knowledge graph construction.")

# ==========================================
# 1. Edge Extraction and Data Definition
# ==========================================
# Defining core biological targets, drug compounds, and clinical ADR outcomes.
# These represent validated high-weight associations extracted from the 2026 baseline study.

key_targets = ['HTR1A', 'DRD4', 'HTR2C', 'GABRD', 'GABRR1']

key_drugs = {
    'Fluoxetine': ['HTR1A', 'HTR2C'],
    'Haloperidol': ['DRD4', 'HTR1A'],
    'Diazepam': ['GABRD', 'GABRR1'],
    'Clozapine': ['DRD4', 'HTR2C', 'HTR1A']
}

target_to_adr = {
    'HTR1A': ['Psychiatric Disorders'],
    'DRD4': ['Psychiatric Disorders', 'Nervous System'],
    'HTR2C': ['Psychiatric Disorders', 'Metabolism'],
    'GABRD': ['Nervous System'],
    'GABRR1': ['Nervous System']
}

# ==========================================
# 2. Graph Topology Construction
# ==========================================
print("Constructing tripartite network topology.")
G = nx.DiGraph()

# Node definition with attributes for visualization
# Layer identification for multipartite layout
for drug in key_drugs.keys():
    G.add_node(drug, layer='Drug', color='#27ae60', size=2500)

for target in key_targets:
    G.add_node(target, layer='Target', color='#2980b9', size=1500)

all_adrs = list(set([adr for adrs in target_to_adr.values() for adr in adrs]))
for adr in all_adrs:
    G.add_node(adr, layer='ADR', color='#c0392b', size=3500)

# Edge definition representing validated pharmacological pathways
for drug, targets in key_drugs.items():
    for target in targets:
        G.add_edge(drug, target, weight=2)

for target, adrs in target_to_adr.items():
    for adr in adrs:
        G.add_edge(target, adr, weight=3)

# ==========================================
# 3. Visualization and High-Resolution Export
# ==========================================
print("Rendering network topology.")
plt.figure(figsize=(14, 9), facecolor='white')

# Set multipartite layout parameters
pos = nx.multipartite_layout(G, subset_key="layer", align='horizontal')

# Extract node styling attributes
node_colors = [nx.get_node_attributes(G, 'color')[node] for node in G.nodes()]
node_sizes = [nx.get_node_attributes(G, 'size')[node] for node in G.nodes()]

# Drawing network components
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, edgecolors='white', linewidths=2)
nx.draw_networkx_edges(G, pos, edge_color='#7f8c8d', width=2.5, arrowsize=20, arrowstyle='->', alpha=0.6)

labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=11, font_weight='bold', font_color='black', font_family='sans-serif')

# Axis and label formatting for publication
plt.title("Clinical Toxicity Tripartite Knowledge Graph: Drug-Target-ADR Pathways", fontsize=16, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()

# Export figure with high-resolution settings
output_filename = "Toxicity_Knowledge_Graph.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Publication-ready knowledge graph exported to: {output_filename}")
plt.show()