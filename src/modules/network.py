import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def load_model(args):
    """Load specified model for training"""
    
    VAR_LENGTH = len(args.variable_region)

    if args.model == 'MLP': 
        if args.features == 'onehot':
            input_dim = 20 * VAR_LENGTH
            embed = False
        elif args.features == 'continuous':
            input_dim = 128 * VAR_LENGTH
            embed = True
        elif args.features == 'ECFP':
            input_dim = 208 * VAR_LENGTH
            embed = False
        else:
            raise ValueError("Invalid feature set")
        model = MLP(num_in=input_dim, num_inter=128, num_out=1, num_layers=3,  embed=embed, dropout=0.1, var_length=VAR_LENGTH)

    elif args.model == 'CNN':

        if args.features == 'onehot':
            input_dim = 20
            embed = False
        elif args.features == 'continuous':
            input_dim = 128
            embed = True
        elif args.features == 'ECFP':
            input_dim = 208
            embed = False
        else:
            raise ValueError("Invalid feature set")

        model = CNN(num_in=input_dim, conv_dim=128, num_layers=5, dropout=0.2, embed=embed)
    
    elif args.model == 'Transf':
        if args.features == 'onehot':
            input_dim = 20
            embed = False
        elif args.features == 'continuous':
            input_dim = 128
            embed = True
        elif args.features == 'ECFP':
            input_dim = 208
            embed = False
        else:
            raise ValueError("Invalid feature set")

        model = Transformer(input_dim=input_dim, num_layers=1, heads=8, hidden_dim=128, dropout=0.1, embed=embed)

    return model


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class Transformer(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, heads=4, num_layers=3, dropout=0.0, embed=False):
        super().__init__()

        self.embed = embed
        if self.embed:
            self.W_e = nn.Embedding(20, input_dim)

        self.lin = nn.Linear(input_dim, hidden_dim)
        
        # add positional encoding as part of input
        self.positional = PositionalEncoding(d_model=hidden_dim, max_len=6)

        # attn block
        self.attn_block = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_block.append(nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, dropout=dropout))

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        # flatten and output
        self.W_out = nn.Linear(hidden_dim, 1, bias=True)

        self.sig = nn.Sigmoid()

    def forward(self, X):
        
        if self.embed:
            X = self.W_e(X)

        X = self.lin(X)

        # retrieve positional encodings
        X = self.positional(X)

        for attn in self.attn_block:
            attn_out, _ = attn(X, X, X, need_weights=False)
            X = self.norm(X + self.drop(attn_out))

        X = torch.mean(X, dim=1)

        X = self.W_out(X)
        X = self.sig(X)

        return X


class CNN(nn.Module):
    def __init__(self, num_in=20, conv_dim=128, num_layers=3, dropout=0.2, embed=False):
        super().__init__()

        self.embed = embed
        if self.embed:
           self.W_e = nn.Embedding(20, num_in)

        conv_dims = [num_in] + [conv_dim for c in range(num_layers + 1)]
        self.conv_block = nn.ModuleList()
        for n in range(num_layers):
            self.conv_block.append(nn.Conv1d(in_channels=conv_dims[n], out_channels=conv_dims[n+1], kernel_size=2))
            self.conv_block.append(nn.ReLU())
            self.conv_block.append(nn.Dropout(dropout))

        # Flatten and make final prediction
        self.W_out = nn.Linear(conv_dim, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, X):
        
        if self.embed:
            X = self.W_e(X)

        X = X.view(X.shape[0], X.shape[2], X.shape[1])
        for conv in self.conv_block:
            X = conv(X)
            
        X = torch.squeeze(X, dim=-1)

        X = self.W_out(X)
        X = self.sig(X)
        return X


def get_act_fxn(act: str):
    if act == 'relu':
        return F.relu
    elif act == 'gelu':
        return F.gelu
    elif act == 'elu':
        return F.elu
    elif act == 'selu':
        return F.selu
    elif act == 'celu':
        return F.celu
    elif act == 'leaky_relu':
        return F.leaky_relu
    elif act == 'prelu':
        return F.prelu
    elif act == 'silu':
        return F.silu
    elif act == 'sigmoid':
        return nn.Sigmoid()
    

class MLP(nn.Module):
    def __init__(self, num_in, num_inter, num_out, num_layers, 
    act='relu', bias=True, embed=False, dropout=0., var_length=6):
        super().__init__()
        
        # Learnable embedding layer
        self.embed = embed
        if self.embed:
           self.W_e = nn.Embedding(20, num_in // var_length)

        # Linear layers for MLP
        self.W_in = nn.Linear(num_in, num_inter, bias=bias)

        if dropout != 0.:
            print('Dropout: \t', dropout)
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

        self.W_inter = nn.ModuleList([nn.Linear(num_inter, num_inter, bias=bias) for _ in range(num_layers - 2)])
        self.W_out = nn.Linear(num_inter, num_out, bias=bias)
        self.sig = nn.Sigmoid()
        
        # Activation function
        self.act = get_act_fxn(act)
        
    def forward(self, X):

        # flatten inputs as needed
        if len(X.shape) > 2:
            X = X.view(X.shape[0], X.shape[1] * X.shape[2])   

        if self.embed:
            X = self.W_e(X) # [B, 6, 20]
            X = X.view(X.shape[0], X.shape[1] * X.shape[2]) # [B, 120 x 6]
        
        # Embed inputs with input layer
        X = self.act(self.W_in(X))

        # Pass through intermediate layers
        for layer in self.W_inter:
            X = self.act(layer(X))
            if self.drop is not None:
                X = self.drop(X)
            
        # Get output from output layer
        X = self.W_out(X)
        X = self.sig(X)
        return X