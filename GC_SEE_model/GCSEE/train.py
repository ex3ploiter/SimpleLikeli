# -*- coding: utf-8 -*-
"""
@Time: 2023/4/30 16:00 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from sklearn.cluster import KMeans
from GC_SEE_model.GCSEE.model import GCSEE
from GC_SEE_utils import data_processor
from GC_SEE_utils.evaluation import eva
from GC_SEE_utils.result import Result
from GC_SEE_utils.utils import count_parameters, get_format_variables
from torch_geometric.data import Data
from ComputeLikelihood import LikelihoodComputer

from GC_SEE_module.GCN import GCN

from VGAE.VGAE_utils import adj_matrix_from_edge_index
from main_utils import get_VGAE_hidden_models
from VGAE.VGAE_loss import VGAELoss
from VGAE.AutoEncoder_model import GAE



device="cuda" if torch.cuda.is_available() else "cpu"


def train(args, data, logger):
    config = config = {
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "LR": 0.001,
        "EPOCHS": 10 , # Adjust the number of epochs as needed
        "hidden_dim" : 64
    }
    device = config["DEVICE"]
    hidden_dim = config["hidden_dim"]  # Adjust as needed
    
    X=X
    adj=adj

    # Create models
    model=GAE(X.shape[1],config['hidden_dim'])



    # Initialize encoder and decoder
    
    loss_function = VGAELoss()
    

    # Create VGAE model
 

    # Define loss function and optimizer
    
    optimizer = Adam(params=model.parameters(), lr=config["LR"])
    
    for epoch in range(config["EPOCHS"]):
        train_one_epoch(model,X,adj,loss_function,optimizer)    

def train_one_epoch(model,X,adj,loss_function,optimizer):
    model.train()
    model.to(device)
    loss_function.to(device)

    preds = []
    targets = []
    total_loss = 0.

    try:
        adj = adj_matrix_from_edge_index(X, adj)
    except:
        adj=adj

    optimizer.zero_grad()
    
    X_output = model.to(device)(X,adj)
    adj_output = model.to(device)(adj,adj)

    loss = loss_function(adj_output, adj.to(device))+loss_function(X_output, X.to(device))

    total_loss += loss.item()

    loss.backward(retain_graph=True)
    optimizer.step()    




