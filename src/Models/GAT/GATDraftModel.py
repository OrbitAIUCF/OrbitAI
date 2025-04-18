# OrbitAI GNN Model - Rough Draft (Finalized Architecture + Graph Construction)
# Purpose: Classify maneuver decisions from satellite state vectors using GAT-based relational reasoning

import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

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
            self.gat1 = GATConv(in_channels, hidden_dim, heads=2, concat=True, add_self_loops=False, edge_dim=hidden_dim)
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
        # first GAT layer (with or without learned edge embeddings)
        if self.use_edge_embedding and edge_attr is not None:
            edge_embed = self.edge_mlp(edge_attr)
            out, (idx, attn) = self.gat1(
                x, edge_index, edge_attr=edge_embed,
                return_attention_weights=True
            )
        else:
            out, (idx, attn) = self.gat1(
                x, edge_index, return_attention_weights=True
            )

        # store a single 1D tensor of attention scores
        # if multi-head, average across heads
        if attn.dim() == 2:
            attn = attn.mean(dim=0)
        self.attn_weights = attn

        x = nn.functional.relu(out)
        x = self.gat2(x, edge_index)
        return self.classifier(x)
    

    # Hardcoded Attention
    @staticmethod
    def apply_hardcoded_attention(graph_list, proximity_threshold=7000.0, w_vel=0.6, w_dist=0.4):
        """
        For each PyG Data object in graph_list, compute a hardcoded 'attention' score
        based on relative velocity and distance:
        - High rel_vel → higher risk
        - Low distance → higher risk
        attention = (w_vel * rel_vel_norm + w_dist * (1 - dist_norm)) / (w_vel + w_dist)
        
        Adds `data.attn` as a list of floats for each edge.
        """
        for data in graph_list:
            edge_attr = data.edge_attr.numpy()  # shape [E, 2] => [dist, rel_vel]
            dists = edge_attr[:, 0]
            rels = edge_attr[:, 1]
            
            # Normalize features
            max_rel = rels.max() if rels.size > 0 else 1.0
            rel_norm = rels / max_rel
            dist_norm = np.clip(dists / proximity_threshold, 0.0, 1.0)
            
            # Compute risk-based attention
            risk_raw = w_vel * rel_norm + w_dist * (1 - dist_norm)
            data.attn = (risk_raw / (w_vel + w_dist)).tolist()

        return graph_list
    
    
    # Batch load graph version
    @staticmethod
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
    # Export graphs to JSON
    
    @staticmethod
    def export_graph_list_to_json(graph_list, output_dir):
        """
        Exports a list of PyG Data objects into JSON frames.
        Each graph must have either data.attn (hard‑coded) or data.attn_weights (from model).
        """
        os.makedirs(output_dir, exist_ok=True)
        all_frames = []

        for i, data in enumerate(graph_list):
            # 1) figure out which attention list to use
            if hasattr(data, 'attn'):
                attn_list = data.attn
            elif getattr(data, 'attn_weights', None) is not None:
                attn_list = data.attn_weights.tolist()
            else:
                raise ValueError(f"Graph at index {i} has no 'attn' or 'attn_weights'")

            # 2) build node entries
            nodes = [
                {"id": data.sat_ids[n], "x": pos[0], "y": pos[1], "z": pos[2]}
                for n, pos in enumerate(data.x[:, :3].tolist())
            ]

            # 3) build edges with attention
            edges = []
            for (src, tgt), (dist, rel_vel), attn in zip(
                    data.edge_index.t().tolist(),
                    data.edge_attr.tolist(),
                    attn_list
                ):
                edges.append({
                    "source":     src,
                    "target":     tgt,
                    "distance":   dist,
                    "rel_vel":    rel_vel,
                    "attention":  attn
                })

            # 4) assemble frame
            frame = {
                "timestamp": data.timestamp,
                "nodes":     nodes,
                "edges":     edges
            }

            # 5) write it out
            path = os.path.join(output_dir, f"frame_{i:04d}.json")
            with open(path, "w") as f:
                json.dump(frame, f, indent=2)
            all_frames.append(frame)

        # also dump the full sequence
        all_path = os.path.join(output_dir, "all_frames.json")
        with open(all_path, "w") as f:
            json.dump(all_frames, f, indent=2)

        return all_frames


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