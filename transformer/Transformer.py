import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

INPUT_DIM = 9  # Input layer dimensions (time, position, velocity)
EMBED_DIM = 128  # Embedding dimension for input vectors
NUM_HEADS = 12  # Number of attention heads
NUM_LAYERS = 6  # Number of encoder layers
FEED_FORWARD_DIM = 256  # Feedforward layer size
OUTPUT_DIM = 6  # Output dimensions (next state vectors)
SEQ_LENGTH = 270  # Input sequence length
LEARNING_RATE = 0.00001  # Learning rate
BATCH_SIZE = 32  # Batch size
EPOCHS = 50  # Number of training epochs
DROPOUT = 0.1  # Dropout rate

class SatelliteTrajectoryDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.inputs = torch.tensor(self.data.iloc[:, :-6].values, dtype=torch.float32)
        self.outputs = torch.tensor(self.data.iloc[:, -6:].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

def get_dataloaders(csv_file, batch_size=BATCH_SIZE, shuffle=True):
    dataset = SatelliteTrajectoryDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class InputEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(InputEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.linear(x)

class PositionalEncoding(nn.Module):
    def __init__(self, seq_length, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, embed_dim))

    def forward(self, x):
        seq_len = x.size(1)  
        return x + self.pos_embedding[:, :seq_len, :]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return attn_output

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, feed_forward_dim):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, feed_forward_dim)
        self.fc2 = nn.Linear(feed_forward_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()
        
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, feed_forward_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(x,x,x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, feed_forward_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        self_attn_out = self.self_attention(x, x, x)
        x = self.norm1(x + self_attn_out)
        cross_attn_out = self.cross_attention(x, memory, memory)
        x = self.norm2(x + cross_attn_out)
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x

class OutputProjection(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(OutputProjection, self).__init__()
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        return self.fc_out(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, num_heads, feed_forward_dim, num_layers, dropout, seq_len, pred_len):
        super(TransformerModel, self).__init__()
        self.embedding = InputEmbedding(input_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(SEQ_LENGTH, embed_dim)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, feed_forward_dim, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(embed_dim, num_heads, feed_forward_dim, dropout) for _ in range(num_layers)])
        self.output_layer = OutputProjection(embed_dim, output_dim)

    def forward(self, x, tgt):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        tgt = self.embedding(tgt)
        tgt = self.pos_encoding(tgt)
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        for layer in self.decoder_layers:
            tgt = layer(tgt, x)
        
        return self.output_layer(tgt)
