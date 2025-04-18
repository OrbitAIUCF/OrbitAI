# === GRUInfer.py ===
# Autoregressive inference using trained GRU model and exporting with time

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from GRUModel import GRUOrbitDeltaPredictor, GRUOrbitDataset

def autoregressive_gru(model, dataset, initial_seq, steps=180):
    model.eval()
    trajectory = []
    current_seq = torch.tensor(initial_seq, dtype=torch.float32).unsqueeze(0)  # shape: [1, seq_len, 6]

    for _ in range(steps):
        with torch.no_grad():
            delta_norm = model(current_seq).squeeze(0)

        # Predict next normalized state
        next_state_norm = current_seq[0, -1, :] + delta_norm

        # Un-normalize to get true [x, y, z] in km
        next_state = dataset.denormalize(next_state_norm).numpy()
        trajectory.append(next_state[:3])

        # Update sequence with new normalized state
        next_seq = torch.cat([current_seq[0, 1:], next_state_norm.unsqueeze(0)], dim=0)
        current_seq = next_seq.unsqueeze(0)

    return np.array(trajectory)

# === Load model and normalization stats ===
model = GRUOrbitDeltaPredictor()
model.load_state_dict(torch.load("gru_orbit_model.pt"))

stats = np.load("gru_dataset_stats.npz")
mean = stats['mean']
std = stats['std']

# === Load dataset and prepare inference ===
df = pd.read_csv("new_training_data.csv")
df = df[['time','position_x','position_y','position_z','velocity_x','velocity_y','velocity_z']].dropna()
dataset = GRUOrbitDataset(df, sat_ids=None, state_mean=mean, state_std=std)

# === Select test satellite and extract sequence ===
test_id = 49
seq_len = 20
test_df = df.iloc[test_id * 271:(test_id + 1) * 271]
test_state = test_df[['position_x','position_y','position_z','velocity_x','velocity_y','velocity_z']].values
norm_test_state = dataset.normalize(test_state)
initial_seq = norm_test_state[:seq_len]

# === Run prediction ===
future_xyz = autoregressive_gru(model, dataset, initial_seq)

# === Plot predicted vs ground truth orbit ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(test_state[:, 0], test_state[:, 1], test_state[:, 2], label='Ground Truth', color='blue')
ax.plot(future_xyz[:, 0], future_xyz[:, 1], future_xyz[:, 2], label='Predicted', linestyle='dashed', color='orange')
ax.set_title("GRU Autoregressive Orbit Prediction")
ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.set_zlabel("Z (km)")
ax.legend(); plt.show()


# === Save predictions with time ===
start_time = test_df['time'].values[seq_len - 1]  # Start after the 10-step input
future_times = np.arange(start_time + 30, start_time + 30 * (len(future_xyz) + 1), 30)

df_out = pd.DataFrame({
    'time': future_times,
    'position_x': future_xyz[:, 0],
    'position_y': future_xyz[:, 1],
    'position_z': future_xyz[:, 2]
})

df_out.to_csv(f"gru_orbit_predictions_with_time.csv_{test_id}", index=False)
