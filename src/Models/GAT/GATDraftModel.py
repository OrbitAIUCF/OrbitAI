# OrbitAI GNN Model - Rough Draft (Finalized Architecture + Graph Construction)
# Purpose: Classify maneuver decisions from satellite state vectors using GAT-based relational reasoning

import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import random
import modelVisualizer as mv

# -------------------------------
# 1. Synthetic Data Assumption
# -------------------------------
# Each satellite has:
# - 3D Position (x, y, z)
# - 3D Velocity (vx, vy, vz)
# - Time (t)
# Labels: either "maneuver class" or binary collision label

# -------------------------------
# 2. GNN Model Definition
# -------------------------------
class OrbitGNN(nn.Module):
    def __init__(self, in_channels=7, hidden_dim=64, num_classes=4, use_edge_embedding=False):
        super(OrbitGNN, self).__init__()

        self.use_edge_embedding = use_edge_embedding
        self.attn_weights = None # store attention for visualization

        if use_edge_embedding:
            self.edge_mlp = nn.Sequential(
                nn.Linear(2, hidden_dim), # 2 = [distance, relative velocity]
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gat1 = GATConv(in_channels + hidden_dim, hidden_dim, heads=2, concat=True, add_self_loops=False)
            self.gat2 = GATConv(hidden_dim * 2, hidden_dim, heads=2, concat=False, add_self_loops=False)
        else:
            self.gat1 = GATConv(in_channels, hidden_dim, heads=2, concat=True, add_self_loops=False)
            self.gat2 = GATConv(hidden_dim * 2, hidden_dim, heads=2, concat=False, add_self_loops=False)

        # Maneuver decision head (output: maneuver class per node)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)  # e.g., 0 = no move, 1 = +x, 2 = -x, 3 = +z
        )

    def forward(self, x, edge_index, edge_attr=None):
        if self.use_edge_embedding and edge_attr is not None:
            edge_embed = self.edge_mlp(edge_attr) # [num_edges, hidden_dim]
            # Expand node features for each edge
            row = edge_index[0]
            x_src = x[row] # sender
            edge_cat = torch.cat([x_src, edge_embed], dim=1)
            out, attn = self.gat1((x, x), edge_index, return_attention_weights=True, edge_attr=edge_cat)
            self.attn_weights = attn[1] # store attention
            x = out
        else:
            out, attn = self.gat1(x, edge_index, return_attention_weights=True)
            self.attnweights = attn[1]
            x = out

        x = nn.functional.relu(x)
        x = self.gat2(x, edge_index)
        return self.classifier(x)

    def build_graph_from_csv(csv_path, timestamp_str, proximity_threshold=7000.0):
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Convert string timestamp to datetime object
        timestamp = pd.to_datetime(timestamp_str)

        # Filter for a single timestep
        snapshot = df[df["timestamp"] == timestamp]
        snapshot = snapshot.sort_values(by="sat_id").reset_index(drop=True)

        # Encode satellite IDs to index positions
        le = LabelEncoder()
        snapshot["node_idx"] = le.fit_transform(snapshot["sat_id"])

        # Extract features
        node_features = snapshot[["position_x", "position_y", "position_z",
                                "velocity_x", "velocity_y", "velocity_z"]].copy()
        node_features["time"] = timestamp.timestamp()  # add scalar time as final feature
        x = torch.tensor(node_features.values, dtype=torch.float32)

        # Compute edge_index and edge attributes (distance, relative velocity)
        positions = snapshot[["position_x", "position_y", "position_z"]].values
        velocities = snapshot[["velocity_x", "velocity_y", "velocity_z"]].values

        edge_index = [] #Compresssed form of an adjacency matrix.
        edge_features = []
        num_nodes = len(positions)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < proximity_threshold:
                        rel_vel = np.linalg.norm(velocities[i] - velocities[j])
                        edge_index.append([i, j])
                        edge_features.append([dist, rel_vel])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32) #Stores the edge_distances & edge_velocities for use in GAT attention weights

        # Labels (maneuver class)
        y = torch.tensor(snapshot["maneuver_label"].values, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, sat_ids=snapshot["sat_id"].tolist())

    # Batch load graph version
    def build_all_graphs_from_csv(csv_path, proximity_threshold=7000.0):
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        timestamps = sorted(df["timestamp"].unique())
        graph_list = []

        for timestamp in timestamps:
            snapshot = df[df["timestamp"] == timestamp].copy().sort_values(by="sat_id").reset_index(drop=True)
            le = LabelEncoder()
            snapshot["node_idx"] = le.fit_transform(snapshot["sat_id"])

            node_features = snapshot[["position_x", "position_y", "position_z",
                                    "velocity_x", "velocity_y", "velocity_z"]].copy()
            node_features["time"] = timestamp.timestamp()
            x = torch.tensor(node_features.values, dtype=torch.float32)

            positions = snapshot[["position_x", "position_y", "position_z"]].values
            velocities = snapshot[["velocity_x", "velocity_y", "velocity_z"]].values

            edge_index = []
            edge_features = []
            num_nodes = len(positions)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        dist = np.linalg.norm(positions[i] - positions[j])
                        if dist < proximity_threshold:
                            rel_vel = np.linalg.norm(velocities[i] - velocities[j])
                            edge_index.append([i, j])
                            edge_features.append([dist, rel_vel])

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)
            y = torch.tensor(snapshot["maneuver_label"].values, dtype=torch.long)

            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        sat_ids=snapshot["sat_id"].tolist())
            graph.timestamp = str(timestamp)
            graph_list.append(graph)

        return graph_list

# -------------------------------
# 4. Loss and Training Strategy
# -------------------------------
# CrossEntropyLoss for maneuver class prediction
# Synthetic labels generated from heuristics:
#   - E.g., if two objects on collision course, assign maneuver to one
#   - Use direction of approach to assign maneuver class

# -------------------------------
# 5. Output Behavior
# -------------------------------
# For each node:
# - Predict maneuver action class
# - Learn which neighbors to prioritize using GAT attention

# -------------------------------
# 6. Next Steps
# -------------------------------
# [ ] Generate synthetic dataset with 3D positions/velocities + labels
# [ ] Build graph structure + edge_index per timestep
# [ ] Instantiate model and train
# [ ] Evaluate maneuver prediction accuracy

# Note: This model is a rough but practical draft. Further improvements can include:
# - Dynamic graphs (update edges at each step)
# - Edge feature incorporation (e.g., relative velocity or distance)
# - Additional maneuver classes (±vx, vy, etc.)

# End of draft