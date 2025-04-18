# OrbitAI GNN Training Script
# ----------------------------
# Trains OrbitGNN on simulated satellite maneuver labels

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from GATDraftModel import OrbitGNN

# ----------------------------
# Configuration
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# ----------------------------
# Load Dataset
# ----------------------------
graphs = OrbitGNN.build_all_graphs_from_csv("orbits_with_velocity_and_labels.csv")
dataloader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# Model Initialization
# ----------------------------
model = OrbitGNN(use_edge_embedding=True).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# ----------------------------
# Training Loop
# ----------------------------
print("[INFO] Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in dataloader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f}")

# ----------------------------
# Save Model
# ----------------------------
torch.save(model.state_dict(), "orbit_gnn_weights.pth")
print("[INFO] Model saved to orbit_gnn_weights.pth")
