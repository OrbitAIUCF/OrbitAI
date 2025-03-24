import torch.optim as optim
import torch.nn as nn
from OrbitAITransformer import OrbitAI
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time


INPUT_DIM = 7          #The dimensions/neurons for the input layer: time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z
EMBED_DIM = 128        #Embedding Dimension for input vectors.
NUM_HEADS = 8          #Number of attention heads in multi-head attention block
NUM_LAYERS = 8         #Number of encoder layers
FEED_FORWARD_DIM = 512 #Size of feedforward layers within the Transformer's MLP
OUTPUT_DIM = 6         #Predicting the 6 dimensional outputs (the next state vectors)
SEQ_LENGTH = 10        #Length of the input sequences
LEARNING_RATE = 0.0001  #The learning rate for the optimizer function
BATCH_SIZE = 32        #Number of sequences per batch
EPOCHS = 100            #Number of training iterations
DROPOUT = 0.1          #Overfitting prevention
#Add another parameter, dropout, if experiencing overfitting


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        tgt_y = batch['tgt_y'].to(device)

        optimizer.zero_grad()
        output = model(src, tgt)

        loss = criterion(output, tgt_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

class OrbitDataset(Dataset):
    def __init__(self, csv_path, input_len = 20, pred_len = 10):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        #Load the pre-processed CSV (time must be in seconds (float))
        df = pd.read_csv(csv_path)

        #Keep columns in order
        raw_data = df[['time', 'position_x', 'position_y', 'position_z',
                                'velocity_x', 'velocity_y', 'velocity_z']].values
        #Split time and state data
        self.time = raw_data[:, 0] - raw_data[:, 0].min()/(raw_data[:, 0].max() - raw_data[:, 0].min())
        self.time = self.time.reshape(-1,1) #normalize time

        self.state = raw_data[:, 1:]              #Shape [T, 6]

        #Normalize the state vector data
        self.scaler = StandardScaler()
        self.state_scaled = self.scaler.fit_transform(self.state)

        #Reconstruct full data with original time and newly normalized state
        self.data = np.hstack([self.time, self.state_scaled]) #Shape [T, 7]


    def __len__(self):
        return len(self.data) - (self.input_len + self.pred_len) + 1

    def __getitem__(self, idx):
        #Encoder input (10 steps)
        src = self.data[idx : idx + self.input_len]

        #Decoder input (use the last state from src then roll forward)
        tgt = self.data[idx + self.input_len - 1 : idx + self.input_len + self.pred_len - 1]

        #Ground truth for reach decoder step (excluding the time column)
        tgt_y = self.data[idx + self.input_len : idx + self.input_len + self.pred_len, 1:]

        #Convert to tensors and then return
        '''
        src = [10,7]
        tgt = [5,7]
        tgt_y = [5,6]
        '''
        return{
            'src': torch.tensor(src, dtype=torch.float32),
            'tgt': torch.tensor(tgt, dtype=torch.float32),
            'tgt_y': torch.tensor(tgt_y, dtype=torch.float32)
        }

    def inverse_transform(self, prediction):
        #Reverse scaling of predicted state vectors, un-normalized them
        return self.scaler.inverse_transform(prediction)



'''
#Sanity check
batch = next(iter(dataloader))
print("src:   ", batch['src'].shape)    # [BATCH_SIZE, 10, 7]
print("tgt:   ", batch['tgt'].shape)    # [BATCH_SIZE, 5, 7]
print("tgt_y: ", batch['tgt_y'].shape)  # [BATCH_SIZE, 5, 6]
'''

#MODEL TRAINING

#Select Device
device = torch.device('cuda')

#Load Dataset
dataset = OrbitDataset(csv_path = "training_data.csv", input_len = 10, pred_len = 5)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#Instantiate model
MODEL = OrbitAI(
    input_dim = INPUT_DIM,
    embed_dim = EMBED_DIM,
    output_dim = OUTPUT_DIM,
    num_heads = NUM_HEADS,
    feedforward_dim = FEED_FORWARD_DIM,
    num_layers = NUM_LAYERS,
    dropout = DROPOUT,
    seq_len = SEQ_LENGTH
).to(device)

#Select loss function
CRITERION = nn.MSELoss()
#Select optimizer
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE, betas=(0.9,0.98))

#Training Loop
for epoch in range(EPOCHS):
    start_time = time.time()
    MODEL.train()
    total_loss = 0

    avg_loss = train(MODEL, dataloader, OPTIMIZER, CRITERION, device)

    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1:02}/{EPOCHS} | Loss: {avg_loss:.6f} | Time: {elapsed:.2f}")

torch.save({
    'model_state_dict': MODEL.state_dict(),
    'optimizer_state_dict': OPTIMIZER.state_dict()
}, 'orbitai_checkpoint.pth')

print("Model and Optimizer saved")

# Evaluation
MODEL.eval()
with torch.no_grad():
    sample = dataset[0]
    src = sample['src'].unsqueeze(0).to(device)    # [1, input_len, 7]
    tgt = sample['tgt'].unsqueeze(0).to(device)    # [1, pred_len, 7]
    tgt_y = sample['tgt_y']                        # [pred_len, 6]

    output = MODEL(src, tgt).squeeze(0).cpu().numpy()  # [pred_len, 6]
    prediction_unscaled = dataset.inverse_transform(output)
    target_unscaled = dataset.inverse_transform(tgt_y.numpy())

    print("Prediction (unscaled):\n", prediction_unscaled)
    print("Target (unscaled):\n", target_unscaled)

