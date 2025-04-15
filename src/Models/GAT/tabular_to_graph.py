import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load and normalize dataset
df = pd.read_csv("training_data.csv")

features = ["position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z"]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Extract data at a specific timestep
time_step = 0
snapshot = df[df["time"] == time_step].reset_index(drop=True)

# Build k-NN graph
k = 5
nbrs = NearestNeighbors(n_neighbors=k+1).fit(snapshot[features])
_, indices = nbrs.kneighbors(snapshot[features])

edge_index = []
for src, neighbors in enumerate(indices):
    for dst in neighbors[1:]:  # skip self-loop
        edge_index.append([src, dst])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
x = torch.tensor(snapshot[features].values, dtype=torch.float)
graph_data = Data(x=x, edge_index=edge_index)

# Convert to NetworkX for visualization
G = to_networkx(graph_data, to_undirected=True)

# Get 3D positions from normalized data
positions = graph_data.x[:, :3].numpy()
node_colors = graph_data.x[:, 3].numpy()  # Color by velocity_x or another feature

# Build position dict for plotting
pos_3d = {i: positions[i] for i in range(len(positions))}

# --- 3D Visualization ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw edges
for src, dst in G.edges():
    x_vals = [positions[src][0], positions[dst][0]]
    y_vals = [positions[src][1], positions[dst][1]]
    z_vals = [positions[src][2], positions[dst][2]]
    ax.plot(x_vals, y_vals, z_vals, color='gray', linewidth=0.5)

# Draw nodes
sc = ax.scatter(
    positions[:, 0], positions[:, 1], positions[:, 2],
    c=node_colors, cmap='viridis', s=50
)

# Add colorbar
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('velocity_x')

# Axis labels
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("3D Spatial Graph at Time = 0")

plt.tight_layout()
plt.show()
