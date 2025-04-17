# Training File

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from GRUModel import GRUOrbitDeltaPredictor, GRUOrbitDataset

def train_gru_model(model, dataloader, epochs=50):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for x_seq, y_target in dataloader:
            pred = model(x_seq)
            loss = criterion(pred, y_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.10f}")

# === Load data and train model ===
df = pd.read_csv("new_training_data.csv")
df = df[['time','position_x','position_y','position_z','velocity_x','velocity_y','velocity_z']].dropna()

all_sat_ids = list(range(50))
train_ids, _ = train_test_split(all_sat_ids, test_size=0.2, random_state=42)

train_dataset = GRUOrbitDataset(df, sat_ids=train_ids)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = GRUOrbitDeltaPredictor()
train_gru_model(model, train_loader)

# Save model and normalization stats
torch.save(model.state_dict(), "gru_orbit_model.pt")
np.savez("gru_dataset_stats.npz", mean=train_dataset.state_mean, std=train_dataset.state_std)
