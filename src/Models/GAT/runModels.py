from GATDraftModel import OrbitGNN
import modelVisualizer as mv
import torch
import os
import json

# Load graph sequence
graphs = OrbitGNN.build_all_graphs_from_csv("orbits_with_velocity_and_labels.csv")

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"[INFO] Using device: {device}") # Confirm CUDA

#Initialize Model#
model = OrbitGNN(use_edge_embedding=True).to(device).eval()

# Run forward pass per graph and attach attention weights

for graph in graphs:
    row = graph.edge_index[0]
    col = graph.edge_index[1]
    num_nodes = graph.x.size(0)

    # If valid, move to device
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    edge_attr = graph.edge_attr.to(device)

    with torch.no_grad():
        _ = model(x, edge_index, edge_attr=edge_attr)
        graph.attn_weights = model.attn_weights.cpu()


# Visualize in Plotly
mv.animate_graph_3d_interactive(graphs)

#Basic 3D & 2D
# graph = build_graph_from_csv("orbits_with_velocity_and_labels.csv","2025-04-15 16:25:32.375956+00:00")
# mv.visualize_graph(graph)
# mv.visualize_graph_3d(graph)