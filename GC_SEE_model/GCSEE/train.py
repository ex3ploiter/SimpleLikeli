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


def train( data):
    config = config = {
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "LR": 0.1,
        "EPOCHS": 50 , # Adjust the number of epochs as needed
        "hidden_dim" : 64
    }
    device = config["DEVICE"]
    
    
    M = data.M.to(device).float()
    adj_norm = data_processor.normalize_adj(data.adj)
    adj_norm = data_processor.numpy_to_torch(adj_norm).to(device).float()
    adj = data_processor.numpy_to_torch(data.adj).to(device).float()
    adj_label = adj
    feature = data.feature.to(device).float()
    label = data.label    
    
    X=feature
    adj=adj

    # Create models
    model=GAE(X.shape,config['hidden_dim'],clusters=6)



    # Initialize encoder and decoder
    
    loss_function = VGAELoss()
    

    # Create VGAE model
 

    # Define loss function and optimizer
    
    optimizer = Adam(params=model.parameters(), lr=config["LR"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50,gamma=0.1)
    
    for epoch in range(config["EPOCHS"]):
        loss=train_one_epoch(model,X,adj,adj_norm,loss_function,optimizer,scheduler)  
        print(f"VAE Train -- Epoch {epoch}, Loss : {loss}")  

def train_one_epoch(model,X,adj,adj_norm,loss_function,optimizer,scheduler):
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
    criterion2 = torch.nn.MSELoss()
    
    X_output,adj_output,scores = model.to(device)(X,adj)
    
    likelihood=getLikelihood(scores,X, adj, adj_norm)
    likelihood_loss=criterion2(likelihood,torch.tensor([10.],requires_grad=True).to(device))
    
    print('likelihood : ',likelihood)
    print('likelihood_loss : ',likelihood_loss)
    

    # loss = loss_function(adj_output, adj.to(device))+\
    # loss_function(X_output, X.to(device))+\
    # likelihood_loss
    loss=likelihood_loss

    total_loss += loss.item()
    

    loss.backward(retain_graph=True)

  

    optimizer.step()   
    scheduler.step() 
    
    return loss.item()




def getLikelihood(pred,feature, adj, adj_norm):
    
    
    predicted_class = torch.argmax(pred, dim=1)
    clusters = {}

    # Iterate through nodes and assign them to clusters based on predicted labels
    for node, label in enumerate(predicted_class):
        label=label.item()
        if label not in clusters:
            clusters[label] = [node]
        else:
            clusters[label].append(node)    
            
            
    
                    
    Dataset_pyG = Data(x=feature.to(device).float(), edge_index=torch.tensor(adj,dtype=torch.float64,requires_grad=True).nonzero().t().contiguous(),y=torch.tensor(label,dtype=torch.float64,requires_grad=True).cuda()).cuda()
    
    likelihood=0.
    
    for key in clusters.keys():
    
        Cluster=Dataset_pyG.cuda().subgraph(torch.tensor(clusters[key]).cuda())
        model_Likelihood=LikelihoodComputer(Cluster)
        likelihood+=model_Likelihood()
        
    return likelihood