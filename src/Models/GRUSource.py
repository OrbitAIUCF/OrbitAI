import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

# === GRU-BASED ORBIT DELTA PREDICTOR ===
class GRUOrbitDeltaPredictor(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=6, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # x shape: [batch, seq_len, 6]
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])  # Only last output step

# === DATASET CLASS ===
class GRUOrbitDataset(torch.utils.data.Dataset):
    def __init__(self, df, sat_ids, steps=270, sequence_len=20):
        self.sequence_len = sequence_len
        self.samples = []
        self.targets = []

        all_states = []
        for sat_id in sat_ids:
            sample = df.iloc[sat_id * (steps + 1):(sat_id + 1) * (steps + 1)]
            pos = sample[['position_x', 'position_y', 'position_z']].values
            vel = sample[['velocity_x', 'velocity_y', 'velocity_z']].values
            state = np.hstack([pos, vel])
            all_states.append(state)

        full_state = np.vstack(all_states)
        self.state_mean = full_state.mean(axis=0)
        self.state_std = full_state.std(axis=0)

        for state in all_states:
            norm_state = (state - self.state_mean) / self.state_std
            for i in range(len(norm_state) - sequence_len - 1):
                x_seq = norm_state[i:i + sequence_len]
                y_target = norm_state[i + sequence_len + 1] - norm_state[i + sequence_len]
                self.samples.append(x_seq)
                self.targets.append(y_target)

        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

    def denormalize(self, x):
        return x * torch.tensor(self.state_std) + torch.tensor(self.state_mean)

    def normalize_state(self, x):
        return (x - self.state_mean) / self.state_std

# === TRAIN FUNCTION ===
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
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.6f}")

# === AUTOREGRESSIVE INFERENCE ===
def autoregressive_gru(model, dataset, initial_seq, steps=180):
    model.eval()
    trajectory = []
    current_seq = torch.tensor(initial_seq, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, 6]

    for _ in range(steps):
        with torch.no_grad():
            delta_norm = model(current_seq).squeeze(0)
        next_state_norm = current_seq[0, -1, :] + delta_norm
        next_state = dataset.denormalize(next_state_norm).numpy()
        trajectory.append(next_state[:3])
        next_seq = torch.cat([current_seq[0, 1:], next_state_norm.unsqueeze(0)], dim=0)
        current_seq = next_seq.unsqueeze(0)

    return np.array(trajectory)

# === MAIN SCRIPT ===
df = pd.read_csv("new_training_data.csv")
df = df[['time','position_x','position_y','position_z','velocity_x','velocity_y','velocity_z']].dropna()

all_sat_ids = list(range(50))
train_ids, test_ids = train_test_split(all_sat_ids, test_size=0.2, random_state=42)

train_dataset = GRUOrbitDataset(df, sat_ids=train_ids)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

model = GRUOrbitDeltaPredictor()
train_gru_model(model, train_loader)

# === TEST INFERENCE ===
test_id = test_ids[2]
test_df = df.iloc[test_id * 271:(test_id + 1) * 271]
test_state = test_df[['position_x','position_y','position_z','velocity_x','velocity_y','velocity_z']].values

norm_test_state = (test_state - train_dataset.state_mean) / train_dataset.state_std
initial_seq = norm_test_state[:10]  # shape: [10, 6]

future_xyz = autoregressive_gru(model, train_dataset, initial_seq)

# === PLOT ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(
    test_state[:, 0], test_state[:, 1], test_state[:, 2],
    label='Ground Truth', color='blue'
)
ax.plot(
    future_xyz[:, 0], future_xyz[:, 1], future_xyz[:, 2],
    label='Predicted', linestyle='dashed', color='orange'
)
ax.set_title("GRU Autoregressive Orbit Prediction (Normalized)")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.legend()
plt.show()

# === EXPORT ===
pd.DataFrame(future_xyz, columns=['position_x','position_y','position_z']).to_csv("gru_orbit_predictions_normalized.csv", index=False)


