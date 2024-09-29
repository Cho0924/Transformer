import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # make #vocab_size of embeddings with d_model dimension
        # nn.Embedding randomly makes embeddings at first, later on embeddings can be trained
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    # in a subclass of nn.Module, we need to define a forward method which defines how the input flows through the layer
    # forward method will be automatically invoked upon passing an input to the instance (e.g. instance(input))
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

  
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # use log inside exp to avoid number overflowing. 
        # (if we write 10000.0 * arange(0, d_model, 2) directly, the number will become too big when multiplying 10000.0 with big numbers)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
        # Apply the sin to even positions and cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # (1, Seq_len, d_model) add another dimension to accomodate a batch of sentences
        
        self.register_buffer("pe", pe) # pe will be saved fixed in the Module, not getting optimized during traning
        
    def forward(self, x): # x will be an input with size (batch_size, seq_len, d_model)
        x = x + (self.pe[:, x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
        

# In Layer Normalization, every token vector is normalized with its own mean and variance
class LaterNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None: # eps for avoiding division by 0 or a too small number
        super().__init__()
        self.eps = eps
        # alpha is an multiplicative term, beta(bias) is an additive term in the normalization step
        # during traning it will be optimized. Alpha allows the model to scale the outputs at the model's wish
        self.alpha = nn.Parameter(torch.ones(1)) # nn.Parameter is a kind of tensor that is to be considered a module parameter. Will be added to the list of paramerter of the Module
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True) # calculate the mean of #d_model values (-> why dim = -1)
        std = x.std(dim = -1, keepdim=True) # want to keep the dimension so that it is compatible with input x
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
        