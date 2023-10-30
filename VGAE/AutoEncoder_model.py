from typing import Tuple

import torch
from torch import nn
from torch_geometric.data import Data
from GC_SEE_module.GCN import GCN




class GAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.encoder_1=GCN(input_dim, hidden_dim)
        self.encoder_2=GCN(hidden_dim, hidden_dim)
        self.encoder_3=GCN(hidden_dim, hidden_dim//2)
        
        self.decoder_1=GCN(hidden_dim//2, hidden_dim)
        self.decoder_2=GCN(hidden_dim, hidden_dim)
        self.decoder_3=GCN(hidden_dim, input_dim)
        


    def forward(self, X,adj_):
        z=self.encoder_1(X,adj_)
        z=self.encoder_2(z,adj_)
        z=self.encoder_3(z,adj_)
        
        z=self.decoder_1(z,adj_)
        z=self.decoder_2(z,adj_)
        z=self.decoder_3(z,adj_)
        
        
        
        return z
