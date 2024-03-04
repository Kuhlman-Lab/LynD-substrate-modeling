
import torch.nn as nn
import torch.nn.functional as F


def load_model(args):
    """Load specified model for training"""
    
    model = MLP(
        num_in=120, # input features
        num_inter=128, # hidden dim
        num_out=1,  # binary clf has only 1 output
        num_layers=3 # depth (minimum=3)
    )
    return model


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
    def __init__(self, num_in, num_inter, num_out, num_layers, act='relu', bias=True):
        super().__init__()
        
        # Linear layers for MLP
        self.W_in = nn.Linear(num_in, num_inter, bias=bias)
        self.W_inter = nn.ModuleList([nn.Linear(num_inter, num_inter, bias=bias) for _ in range(num_layers - 2)])
        self.W_out = nn.Linear(num_inter, num_out, bias=bias)
        self.sig = nn.Sigmoid()
        
        # Activation function
        self.act = get_act_fxn(act)
        
    def forward(self, X):
        
        # Embed inputs with input layer
        X = self.act(self.W_in(X))

        # Pass through intermediate layers
        for layer in self.W_inter:
            X = self.act(layer(X))
            
        # Get output from output layer
        X = self.W_out(X)
        X = self.sig(X)
        return X