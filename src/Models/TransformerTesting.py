from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from OrbitAITransformer import OrbitAI
from TransformerTraining import OrbitDataset
import torch

#-----------------------------------------------------------------------------------------------------------------------
INPUT_DIM = 7          #The dimensions/neurons for the input layer: time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z
EMBED_DIM = 128        #Embedding Dimension for input vectors.
NUM_HEADS = 8          #Number of attention heads in multi-head attention block
NUM_LAYERS = 6         #Number of encoder layers
FEED_FORWARD_DIM = 256 #Size of feedforward layers within the Transformer's MLP
OUTPUT_DIM = 6         #Predicting the 6 dimensional outputs (the next state vectors)
LEARNING_RATE = 0.00001  #The learning rate for the optimizer function
BATCH_SIZE = 32        #Number of sequences per batch
EPOCHS = 50            #Number of training iterations
DROPOUT = 0.1          #Overfitting prevention
#Add another parameter, dropout, if experiencing overfitting
SEQ_LENGTH = 270        #Length of the input sequences, 270 = 8100/30 = PropagationDuration/steps
PRED_LEN = 2           #Number of sequences we want outputted
#-----------------------------------------------------------------------------------------------------------------------

#Load Dataset
dataset = OrbitDataset(csv_path = "training_data.csv", input_len = 270, pred_len = 1)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
device= torch.device('cuda')

#Instantiate model
MODEL = OrbitAI(
    input_dim = INPUT_DIM,
    embed_dim = EMBED_DIM,
    output_dim = OUTPUT_DIM,
    num_heads = NUM_HEADS,
    feedforward_dim = FEED_FORWARD_DIM,
    num_layers = NUM_LAYERS,
    dropout = DROPOUT,
    seq_len = SEQ_LENGTH,
    pred_len = PRED_LEN
).to(device)

state_dict = torch.load('orbitai_checkpoint.pth',map_location=device)
MODEL.load_state_dict(state_dict['model_state_dict'])

# Evaluation
MODEL.eval()
with torch.no_grad():
    sample = dataset[0]
    src = sample['src'].unsqueeze(0).to(device)    # [1, input_len, 7] this is the input sequence we want the model to learn from, historical trajectory data. When src is encoded it becomes memory
    tgt = sample['tgt'].unsqueeze(0).to(device)    # [1, pred_len, 7] this is the input to the decoder to start generating predictions, the previously known state
    tgt_y = sample['tgt_y']                        # [pred_len, 6]

    output = MODEL(src, tgt).squeeze(0).cpu().numpy()  # [pred_len, 6]
    prediction_unscaled = dataset.inverse_transform(output)
    target_unscaled = dataset.inverse_transform(tgt_y.numpy())

    print("Prediction:\n", prediction_unscaled[:5])  # Check if it's real values
    print("Target:\n", target_unscaled[:5])

    time = list(range(target_unscaled.shape[0]))

    print("Time length:", len(time))
    print("Target shape:", target_unscaled.shape)
    print("Prediction shape:", prediction_unscaled.shape)


    # target_unscaled and prediction_unscaled are shape [pred_len, 6]

    for i, label in enumerate(["X", "Y", "Z"]):
        plt.figure()
        plt.plot(time, target_unscaled[:, i], label=f"True{label}")
        plt.plot(time, prediction_unscaled[:, i], label=f"Predicted{label}")
        plt.xlabel("Timestep")
        plt.ylabel(f"Position {label} (km)")
        plt.title(f"{label}-Axis Position: True vs. Predicted")
        plt.legend()
        plt.grid(True)
        plt.show()
