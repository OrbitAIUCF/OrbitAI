# Gated Recurrent Network

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn

# GRU-based delta predictor for orbital states
class GRUOrbitDeltaPredictor(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=6, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])  # Output only the last time step

# Custom Dataset for GRU orbit prediction
class GRUOrbitDataset(Dataset):
    def __init__(self, df, sat_ids=None, steps=270, sequence_len=20, state_mean=None, state_std=None):
        self.sequence_len = sequence_len
        self.samples = []
        self.targets = []

        all_states = []
        if sat_ids is not None:
            # Training mode: compute stats from training data
            for sat_id in sat_ids:
                sample = df.iloc[sat_id * (steps + 1):(sat_id + 1) * (steps + 1)]
                pos = sample[['position_x', 'position_y', 'position_z']].values
                vel = sample[['velocity_x', 'velocity_y', 'velocity_z']].values
                state = np.hstack([pos, vel])
                all_states.append(state)

            full_state = np.vstack(all_states)
            self.state_mean = full_state.mean(axis=0) if state_mean is None else state_mean
            self.state_std = full_state.std(axis=0) if state_std is None else state_std

            for state in all_states:
                norm_state = (state - self.state_mean) / self.state_std
                for i in range(len(norm_state) - sequence_len - 1):
                    x_seq = norm_state[i:i + sequence_len]
                    y_target = norm_state[i + sequence_len + 1] - norm_state[i + sequence_len]  # Δ prediction
                    self.samples.append(x_seq)
                    self.targets.append(y_target)

            self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
            self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)
        else:
            # Inference mode: mean/std must be supplied
            self.state_mean = state_mean
            self.state_std = state_std

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx], self.targets[idx]
    def denormalize(self, x): return x * torch.tensor(self.state_std) + torch.tensor(self.state_mean)
    def normalize(self, x): return (x - self.state_mean) / self.state_std
