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
    def __init__(self, in_channels=7, hidden_dim=64, num_classes=4):
        super(OrbitGNN, self).__init__()

        self.gat1 = GATConv(in_channels, hidden_dim, heads=2, concat=True)
        self.gat2 = GATConv(hidden_dim * 2, hidden_dim, heads=2, concat=False)

        # Maneuver decision head (output: maneuver class per node)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)  # e.g., 0 = no move, 1 = +x, 2 = -x, 3 = +z
        )

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = nn.functional.relu(x)
        x = self.gat2(x, edge_index)
        return self.classifier(x)

    # -------------------------------
    # 3. Graph Construction Logic
    # -------------------------------
    # (To be implemented with a separate utility)
    # - Compute edges where distance(position_i - position_j) < proximity_threshold
    # - For each edge, optionally compute attention weight from relative velocity
    # - Encode edge weights if needed (for attention override)

    def build_graph_from_csv(csv_path, timestamp_str, proximity_threshold=10000.0):
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

        # Compute edge_index and edge attributes (distance)
        positions = snapshot[["position_x", "position_y", "position_z"]].values
        edge_index = [] #Compresssed form of an adjacency matrix.
        edge_distances = []
        num_nodes = len(positions)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < proximity_threshold:
                        edge_index.append([i, j])
                        edge_distances.append(dist)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_distances, dtype=torch.float32) #Stores the edge_distances for potential use in GAT attention weights

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
            edge_index = []
            edge_distances = []
            num_nodes = len(positions)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        dist = np.linalg.norm(positions[i] - positions[j])
                        if dist < proximity_threshold:
                            edge_index.append([i, j])
                            edge_distances.append(dist)

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_distances, dtype=torch.float32)
            y = torch.tensor(snapshot["maneuver_label"].values, dtype=torch.long)

            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        sat_ids=snapshot["sat_id"].tolist())
            graph.timestamp = str(timestamp)
            graph_list.append(graph)

        return graph_list

    #Run Visualization Model#
    
    #Ineractive 3D
    graphs = build_all_graphs_from_csv("orbits_with_velocity_and_labels.csv")
    mv.animate_graph_3d_interactive(graphs)

    #Basic 3D & 2D
    # graph = build_graph_from_csv("orbits_with_velocity_and_labels.csv","2025-04-15 16:25:32.375956+00:00")
    # mv.visualize_graph(graph)
    # mv.visualize_graph_3d(graph)

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
