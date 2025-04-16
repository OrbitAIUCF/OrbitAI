import torch
import pandas as pd
import numpy as np
from OrbitAITransformer import OrbitAI
from TransformerTraining import OrbitDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from src.Models.TransformerTraining import FEED_FORWARD_DIM

# Hyperparameters
CSV_PATH = "new_training_data.csv"
MODEL_PATH = "optimal_weights_orbitai.pth"
SEQ_LENGTH = 120 #What the encoder was trained on
PRED_LEN = 90   #a full orbit in LEO
INPUT_DIM = 7
OUTPUT_DIM = 6
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 4
FEEDFORWARD_DIM = 256
DROPOUT = 0.4




'''
This function will do the following: Autoregression
1. Feed the encoder the full input sequence (src)
2. Feed the decoder only the last step of the encoder (tgt)
3. Get a prediction and append it to the decoder input
4. Repeat step 2 using the predictions instead of the ground truth
'''

def autoregressive_predict(model, src, steps, device):
    model.eval()
    src = src.to(device) #[1,seq_len,7]

    with torch.no_grad():
        #Initial decoder input is the last frame of src
        last_frame = src[:, -1:, :] #[1,1,7]
        decoder_input = last_frame.clone()

        outputs = []

        for _ in range(steps):
            #Predict next state
            out = model(src, decoder_input) #[1,cur_len, 6]
            next_pred = out[:,-1:,:] #Use only the last predicted step

            #Convert it ot full input dim
            next_input = torch.zeros((1,1,7),device = device)

            last_state = decoder_input[:, -1:, 1:] #Last full state (pos+vel)
            new_state = last_state + next_pred
            next_input[:,:,1:] = new_state #use as next full input


            #And keep time stepping forward
            next_input[:,:,0] = decoder_input[:,-1:,0] + delta_t #Advance time or try to use real time

            decoder_input = torch.cat([decoder_input, next_input], dim=1)
            outputs.append(next_pred.squeeze(0).cpu().numpy()) #[1,6] becomes [6]
    #[steps,6]
    return np.array(outputs)


device = torch.device("cuda")

#Load the model
model = OrbitAI(
    input_dim=INPUT_DIM,
    embed_dim=EMBED_DIM,
    output_dim=OUTPUT_DIM,
    num_heads=NUM_HEADS,
    feedforward_dim=FEEDFORWARD_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    seq_len=SEQ_LENGTH,
    pred_len=PRED_LEN
).to(device)

model.load_state_dict(torch.load(MODEL_PATH))
print("Model Loaded.")

#Load dataset for scalers
#This will give us access to time/position/velocity scalers and raw ground truth
dataset = OrbitDataset(CSV_PATH, input_len=SEQ_LENGTH, pred_len=PRED_LEN, split='val')

#4/10/2025 CHANGE 001
delta_t = dataset.normalized_time_step

#Select a sequence
sample = dataset[0] #Choose first validation sequence
src = sample['src'].unsqueeze(0).to(device) #[1,270,7]
tgt = sample['tgt'].unsqueeze(0).to(device) #[1,180,7]
tgt_y = sample['tgt_y'].unsqueeze(0)        #[1,180,6]

#Run inference:
output_np = autoregressive_predict(model, src,steps=PRED_LEN, device=device)
tgt_y_np = tgt_y.squeeze(0).numpy()         #[180,6]


#----------------------------------------------------------------------------------------------------------
#Un-normalize predictions and ground truth, get back to physical units
predicted_deltas_unscaled  = dataset.inverse_transform_deltas(output_np.squeeze(1))

#4/10/2025 Change #002
#Convert deltas to absolute positions
initial_prediction_state = sample['tgt'][0,1:].numpy() #Unscaled initial state vector
initial_prediction_state = np.hstack([
    dataset.position_scaler.inverse_transform(initial_prediction_state[:3].reshape(1, -1)).flatten(),
    dataset.velocity_scaler.inverse_transform(initial_prediction_state[3:].reshape(1, -1)).flatten()
])

trajectory = [initial_prediction_state]
for delta in predicted_deltas_unscaled:
    next_state = trajectory[-1] + delta
    trajectory.append(next_state)

predictions = np.array(trajectory[1:]) #Discard the seed step

#----------------------------------------------------------------------------------------------------------
#4/10/2025 Change #002
#Inverse transform deltas for ground truth values as well
#print("[PRE] tgt_y_np[:3]:", tgt_y_np[:3])  # Should be tiny (~0.001)
true_deltas_unscaled = dataset.inverse_transform_deltas(tgt_y_np)
#print("[POST] true_deltas_unscaled[:3]:", true_deltas_unscaled[:3])  # Should now be large (10s to 100s)

# Extract normalized values
initial_raw = sample['tgt'][0].numpy()
position_norm = initial_raw[1:4].reshape(1, -1)
velocity_norm = initial_raw[4:7].reshape(1, -1)

# Inverse transform both
initial_position = dataset.position_scaler.inverse_transform(position_norm).flatten()
initial_velocity = dataset.velocity_scaler.inverse_transform(velocity_norm).flatten()

# Concatenate into full state vector in real units
initial_ground_truth = np.hstack([initial_position, initial_velocity])

# Roll forward using true deltas
ground_truth = [initial_ground_truth]
for delta in true_deltas_unscaled:
    next_state = ground_truth[-1] + delta
    ground_truth.append(next_state)
ground_truth = np.array(ground_truth[1:])  # drop initial seed

#----------------------------------------------------------------------------------------------------------
#Debugging
print("Initial Position (after inverse):", initial_position)
print("First Delta (position):", true_deltas_unscaled[0, :3])
print("First Step (expected):", initial_position + true_deltas_unscaled[0, :3])

#Plot the orbits
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

#Trajectory Plotting

ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], label='Ground Truth', linewidth=2)
ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], label='Predicted', linestyle='--')

ax.set_xlabel('X Position (km)')
ax.set_ylabel('Y Position (km)')
ax.set_zlabel('Z Position (km)')
ax.set_title('OrbitAI: Predicted vs Ground Truth Orbit')
ax.legend()
plt.show()


#Per-step delta error plotting
'''
plt.plot(true_deltas_unscaled[:, 0], label='True ΔX')
plt.plot(predicted_deltas_unscaled[:, 0], label='Predicted ΔX')
plt.title("Delta Position X over Time")
plt.legend()
plt.show()'''


for i, label in enumerate(['X', 'Y', 'Z']):
    rmse = np.sqrt(mean_squared_error(ground_truth[:, i], predictions[:, i]))
    print(f"Position {label} RMSE: {rmse:.4f} km")

for i, label in enumerate(['Vx', 'Vy', 'Vz']):
    rmse = np.sqrt(mean_squared_error(ground_truth[:, i + 3], predictions[:, i + 3]))
    print(f"Velocity {label} RMSE: {rmse:.4f} km/s")


print(f"Ground Truth Values for Sat 1: {ground_truth[0]}\n, {ground_truth[1]}\n, {ground_truth[2]}")
print(f"Predicted Values for Sat 1: {predictions[0]}\n, {predictions[1]}\n, {predictions[2]}")

