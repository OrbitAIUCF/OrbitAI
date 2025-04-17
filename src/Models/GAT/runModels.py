from GATDraftModel import OrbitGNN  # or wherever your model class is
import modelVisualizer as mv
import torch


# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}") # Confirm CUDA

# Load graph sequence
graphs = OrbitGNN.build_all_graphs_from_csv("orbits_with_velocity_and_labels.csv")

#Initialize Model#
model = OrbitGNN(use_edge_embedding=True).to(device)
model.eval()

# Run forward pass per graph and attach attention weights
for graph in graphs:
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    edge_attr = graph.edge_attr.to(device)

    with torch.no_grad():
        _ = model(x, edge_index, edge_attr=edge_attr)
        graph.attn_weights = model.attn_weights.cpu()  # Store for 3D viewer

# Visualize in Plotly
mv.animate_graph_3d_interactive(graphs)

#Basic 3D & 2D
# graph = build_graph_from_csv("orbits_with_velocity_and_labels.csv","2025-04-15 16:25:32.375956+00:00")
# mv.visualize_graph(graph)
# mv.visualize_graph_3d(graph)