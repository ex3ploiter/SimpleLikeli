from typing import Tuple

import torch
from torch import nn
from torch_geometric.data import Data
from GC_SEE_module.GCN import GCN




class GAE(nn.Module):
    def __init__(self, input_dim, hidden_dim,clusters):
        super().__init__()

        self.encoder_1=GCN(input_dim[1], hidden_dim)
        self.encoder_2=GCN(hidden_dim, hidden_dim)
        self.encoder_3=GCN(hidden_dim, hidden_dim//2)
        
        self.decoder_1_X=GCN(hidden_dim//2, hidden_dim)
        self.decoder_2_X=GCN(hidden_dim, hidden_dim)
        self.decoder_3_X=GCN(hidden_dim, input_dim[1])
        
        
        self.decoder_1_adj=GCN(hidden_dim//2, hidden_dim)
        self.decoder_2_adj=GCN(hidden_dim, hidden_dim)
        self.decoder_3_adj=GCN(hidden_dim, input_dim[0])

        
        
        
        self.mlp1=nn.Linear(hidden_dim//2,clusters)
        


    def forward(self, X,adj_):
        
        z=self.encoder_1(X,adj_)    
        z=self.encoder_2(z,adj_)
        z=self.encoder_3(z,adj_)


        
        z_X=self.decoder_1_X(z,adj_)
        z_X=self.decoder_2_X(z_X,adj_)
        z_X=self.decoder_3_X(z_X,adj_)
    
        z_adj=self.decoder_1_adj(z,adj_)
        z_adj=self.decoder_2_adj(z_adj,adj_)
        z_adj=self.decoder_3_adj(z_adj,adj_)
        
        probabilities = torch.softmax(self.mlp1(z), dim=1)

    
        
        
        return z_X,z_adj,probabilities
