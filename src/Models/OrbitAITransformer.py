import torch
import torch.nn as nn

#Hyperparameters:
'''
Explanations (as needed):

    EMBED_DIM: the embedding dimension is the size of the transformed input features before they
    pass through the attention layers.
    
    Having a higher dimension allows the transformer to capture more nuanced interactions and relationships
    between values in the input sequence. 128 dimensions is a common choice, balancing learning accuracy and compute
    
    FEED_FORWARD_DIM: the number of layers in the MLP (feed-forward network) is commonly between 2-4 larger than the
    embedding layer. Increasing the dimensions increases complexity and learning power.
    
     
'''

'''
Embedding Layer: Since our input consists of 6 continuous features:
(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z) we will project this into a higher-dimensional space using a fully connected
layer.
This will help the model learn feature representations that capture relationships between input variables.

'''
class InputEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(InputEmbedding, self).__init__() #initializes the parent module (nn.Module)
        self.linear = nn.Linear(input_dim, embed_dim) #Linear layer to project input into embedding space.

    def forward(self, x):
        return self.linear(x)

'''
Positional Encoding Layer (Based on time stamps)
'''
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, seq_length, embed_dim):
        super(LearnedPositionalEncoding, self).__init__()
        #One positional vector per sequence position
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, embed_dim))

    def forward(self, x):
        '''
        x:Tensor of shape (BATCH_SIZE, SEQ_LENGTH, EMBED_DIM)
        '''
        seq_length = x.size(1)

        #By slicing with the following code, we can select embeddings that exactly match the current input sequence length
        x = x + self.pos_embedding[:, :seq_length, :]
        return x
'''
Transformer Encoder

For both the Encoder and Decoder, the built-in PyTorch model includes:
Multi-head self-attention
Layer Norm
Skip Connections
Feed Forward MLPs
'''
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, num_layers,dropout):
        super(TransformerEncoder, self).__init__()

        #Single encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            dim_feedforward = feedforward_dim,
            dropout = dropout,
            batch_first = True #PyTorch expects [sequence_length, batch, embed] shaped tensor by default
        )

        #Now to create a stack of encoder layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        '''
        src: input data, tensor of shape (batch_size, sequence_length, embedding_dimensions)
        returns: (batch_size, sequence_length, embedding_dimension)
        '''

        #Pass through transformer encoder for prediction
        encoded_data = self.encoder(src)

        return encoded_data
'''
Transformer Decoder
'''
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, num_layers, dropout):
        super(TransformerDecoder, self).__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            dim_feedforward = feedforward_dim,
            dropout = dropout,
            batch_first = True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = num_layers)

    def forward(self, tgt, memory):
        '''

        tgt: a typical name for the input sequence being sent into a decoder, short for target
        memory: (batch, src_sequence_length, embed_dim) - this is output from the encoder
        '''
        tgt = tgt
        memory = memory

        #Decode!
        out = self.decoder(tgt=tgt, memory=memory)

        return out
'''
Output Layer
input shape: [batch_size, sequence_length, embedding_dimensions]
output shape: [batch_size, sequence_length, 6] for the future state vectors
'''
class OutputProjection(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(OutputProjection,self).__init__()
        #Fully connected MLP layer
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self,x):
        return self.fc_out(x)

'''
OrbitAI Transformer Model
'''
class OrbitAI(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, num_heads, feedforward_dim, num_layers, dropout, seq_len, pred_len):
        super(OrbitAI, self).__init__()

        self.embedding = InputEmbedding(
            input_dim = input_dim,
            embed_dim = embed_dim
            )
        self.src_encoded = LearnedPositionalEncoding(seq_len, embed_dim)
        self.tgt_encoded = LearnedPositionalEncoding(pred_len, embed_dim)

        self.encoder = TransformerEncoder(
            embed_dim = embed_dim,
            num_heads = num_heads,
            feedforward_dim = feedforward_dim,
            num_layers = num_layers,
            dropout = dropout
        )

        self.decoder = TransformerDecoder(
            embed_dim = embed_dim,
            num_heads = num_heads,
            feedforward_dim = feedforward_dim,
            num_layers = num_layers,
            dropout = dropout
        )

        self.output_layer = OutputProjection(
            embed_dim = embed_dim,
            output_dim = output_dim
        )

    def forward(self, src, tgt):
        '''
        src: [batch_size, src_seq_len, input_dim] (from the encoder)
        tgt: [batch_size, tgt_seq_len, input_dim] (from the decoder)
        '''
        src_embedded = self.embedding(src)
        src_encoded = self.src_encoded(src_embedded)

        tgt_embedded = self.embedding(tgt)
        tgt_encoded = self.tgt_encoded(tgt_embedded)

        #Transformer encoder
        memory = self.encoder(src_encoded)

        #Transformer decoder
        decoded = self.decoder(tgt_encoded, memory)

        #Project decoder output into state vector predictions
        output = self.output_layer(decoded)

        return output

'''
Testing the model so far:

#Get input
df = pd.read_csv("training_data.csv")


#Convert from df to Tensor
sequence = df[['time',
               'position_x', 'position_y', 'position_z',
               'velocity_x', 'velocity_y', 'velocity_z'
               ]].values

scaler = StandardScaler()
scaled_data = scaler.fit_transform(sequence)
# After training, you'll need to use `scaler.inverse_transform` to get physical units again

#Convert to float32 tensor and add batch dimension
sequence = torch.tensor(sequence,dtype=torch.float32)

input_seq = sequence[:SEQ_LENGTH, :]             #Shape: [10, 7]
target_state = sequence[SEQ_LENGTH, 1:]          #Shape [6] -> pos and vel, skip the time feature

#Add batch dimension: [1, 10, 7]
input_seq = input_seq.unsqueeze(0)

#Instantiate the model
embed_layer = InputEmbedding(INPUT_DIM, EMBED_DIM)
pos_encoder = LearnedPositionalEncoding(SEQ_LENGTH, EMBED_DIM)
transformer_encoder = TransformerEncoder(
    embed_dim = EMBED_DIM,
    num_heads = NUM_HEADS,
    feedforward_dim = FEED_FORWARD_DIM,
    num_layers = NUM_LAYERS,
    dropout = DROPOUT
)

#Forward pass through each layer
x = embed_layer(input_seq)
x = pos_encoder(x)
x = transformer_encoder(x)
print("After transformer Encoder:", x.shape)

#Use the last step of the encoder as decoder input
tgt = x[:, -1:, :] #[1,1,128]

#Testing the decoder
decoder = TransformerDecoder(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    feedforward_dim=FEED_FORWARD_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

memory = transformer_encoder(x) #Shape [1,1,128], this is the output from the encoder

decoder_output = decoder(tgt=tgt, memory=memory) #[1,1,128]

output_layer = nn.Linear(EMBED_DIM, OUTPUT_DIM)
predicted_state = output_layer(decoder_output) #[1,1,6]

print("Predicted state:", predicted_state)
print("Target state:", target_state)
'''
