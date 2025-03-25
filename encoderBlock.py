import torch as nn
import numpy


INPUT_DIM = 7          #The dimensions/neurons for the input layer: time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z
EMBED_DIM = 128        #Embedding Dimension for input vectors.
NUM_HEADS = 12          #Number of attention heads in multi-head attention block
NUM_LAYERS = 6         #Number of encoder layers
FEED_FORWARD_DIM = 256 #Size of feedforward layers within the Transformer's MLP
OUTPUT_DIM = 6         #Predicting the 6 dimensional outputs (the next state vectors)
SEQ_LENGTH = 10        #Length of the input sequences
LEARNING_RATE = 0.001  #The learning rate for the optimizer function
BATCH_SIZE = 32        #Number of sequences per batch
EPOCHS = 50            #Number of training iterations
DROPOUT = 0.1          #Overfitting prevention
#Add another parameter, dropout, if experiencing overfitting


# Nickie: Transformer Encoder Block

'''
Self Attention Layers + Feed Forward Layers + Add & Norm Layers
Repeat the encoder block 6 times
'''

class EncoderBlock(nn.Module):
    def __init(self, embed_dim, heads, feed_forward, dropout):
      
      super(EncoderBlock, self).__init__()

      self.layers = nn.ModuleList([EncoderBlock(embed_dim, heads, feed_forward, dropout)
      for i in range(NUM_LAYERS)
      ])
      # initializes the layers

      # Multi-Head Attention
      self.attention = nn.MultiheadAttention(embed_dim, heads)

      # Add & Norm
      self.norm1 = nn.LayerNorm(embed_dim)
      self.norm2 = nn.LayerNorm(embed_dim)

      # Feed-Forward Network (MLP)
      self.fc1 = nn.Linear(embed_dim, feed_forward)
      self.fc2 = nn.Linear(feed_forward, embed_dim)

      # Dropout for regularization
      self.relu = nn.ReLU()
      self.dropout = nn.Dropout(dropout)

    # Self-Attention Block

    def forwardMethod(self, x): 

      # perform positional encoding
      x = self.postional_encoding(x)
      
      # stack the layers
      for i in self.layers:
        x = layer(x)

      # Self-Attention Block
      attention_out, _ = self.attention(x,x,x) # Self-attention (query, key, value are all x)
      x = x + attention_out # skip connection
      x = self.norm1(x) # LayerNorm

      # feed-forward network block
      ffn_out = self.fc1(x) # First Linar Layer
      ffn_out = self.relu(ffn_out) # Activation
      ffn_out - self.fc2(ffn_out) # Second Linear Layer
      ffn_out = self.dropout(ffn_out) # dropout
      x = x + ffn_out
      x = self.norm2(x)

      return x
      
