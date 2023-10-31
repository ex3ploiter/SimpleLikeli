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
from VGAE.VGAE_New import VGAE_New






device="cuda" if torch.cuda.is_available() else "cpu"

import matplotlib.pyplot as plt

def visualize_loss_and_accuracy(loss_history, accuracy_history):
    plt.figure(figsize=(12, 5))
    plt.rcParams['font.family'] = 'serif'


    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)
    plt.title('Training NMI')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    # plt.show()
    plt.savefig('./plot.png')
    plt.close()


def train( args,data):
    config = config = {
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "LR": 0.1,
        "EPOCHS": 200 , # Adjust the number of epochs as needed
        "hidden_dim" : 128
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
    model=GAE(X.shape,config['hidden_dim'],clusters=args.clusters)



    # Initialize encoder and decoder
    
    loss_function = VGAELoss()
    

    # Create VGAE model
 

    # Define loss function and optimizer
    
    optimizer = Adam(params=model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100,gamma=0.1)
    
    loss_history = []
    accuracy_history = []    
    
    for epoch in range(config["EPOCHS"]):
        model.train()
        loss=train_one_epoch(model,X,adj,adj_norm,loss_function,optimizer,scheduler)  
        print(f"VAE Train -- Epoch {epoch}, Loss : {loss}")  
        
        model.eval()
        _,_,prediction = model.to(device)(X,adj)
        y_pred = prediction.data.cpu().numpy().argmax(1)
        acc, nmi, ari, f1 = eva(label, y_pred)
        print(f'epoch={epoch:0>3d}, acc={acc:0>.4f}, nmi={nmi:0>.4f}, ari={ari:0>.4f}, f1={f1:0>.4f}')
        
        loss_history.append(loss)
        accuracy_history.append(nmi)
        
        visualize_loss_and_accuracy(loss_history,accuracy_history)
        
        

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
    
    
    X_output,adj_output,prediction = model.to(device)(X,adj)
    
    

    temp_loss=0.
    for i in range(prediction.shape[1]):
        scores=prediction[:,i]
        feature_temp=X*scores[:, None]
        
        adj_temp=adj*scores
        adj_temp=adj_temp*scores
        
        adj_norm_temp=adj_norm*scores
        adj_norm_temp=adj_norm_temp*scores                
        
 
        
        temp_loss+=getLikelihood_temp(feature_temp,adj,adj_norm_temp)    
    
    # likelihood=getLikelihood(scores,X, adj, adj_norm)
    # likelihood_loss=criterion2(likelihood,torch.tensor([10.],requires_grad=True).to(device))
    
    # print('likelihood : ',likelihood)
    # print('likelihood_loss : ',likelihood_loss)
    

    loss = 2*loss_function(adj_output, adj.to(device))+\
    loss_function(X_output, X.to(device))+\
        10*temp_loss
    # likelihood_loss
    # loss=temp_loss

    total_loss += loss.item()
    

    loss.backward(retain_graph=True)

  

    optimizer.step()   
    scheduler.step() 
    
    return loss.item()




def getLikelihood_temp(feature, adj, adj_norm):
    
    
    model=VGAE_New(64,feature.shape[1])
    
    model_temp=LikelihoodComputer(feature,adj_norm,model)
    return model_temp()
    
    
    