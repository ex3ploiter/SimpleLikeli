import torch
from torch import nn


class VGAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    
        self.ce = nn.CrossEntropyLoss()
        

    def forward(self, adj_output, adj_target):
        cross_entropy = self.norm * self.ce(adj_output.flatten(), adj_target.flatten())
        
        loss=cross_entropy
        return loss