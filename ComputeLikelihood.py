from VGAE.VGAE_model import VGAE, Encoder, Decoder
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import Subset

from tqdm.notebook import tqdm

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Sequential, GCNConv

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from VGAE.VGAE_utils import adj_matrix_from_edge_index
from main_utils import get_VGAE_hidden_models
from VGAE.VGAE_loss import VGAELoss


class LikelihoodComputer(nn.Module):
    def __init__(self, X,adj):
        super(LikelihoodComputer, self).__init__()
        self.config = config = {
            "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
            "LR": 0.001,
            "EPOCHS": 10 , # Adjust the number of epochs as needed
            "hidden_dim" : 64
        }
        self.device = self.config["DEVICE"]
        self.hidden_dim = self.config["hidden_dim"]  # Adjust as needed
        
        self.X=X
        self.adj=adj

        # Create models

        self.hidden_model, self.mean_model, self.std_model = get_VGAE_hidden_models(
            self.X, self.hidden_dim)

        # Initialize encoder and decoder
        self.encoder = Encoder(
            hidden_model=self.hidden_model,
            mean_model=self.mean_model,
            std_model=self.std_model,
            hidden_dim=self.hidden_dim,
            input_dim=self.X.shape[1]
            
        )

        self.decoder = Decoder()

        # Create VGAE model
        self.model = VGAE(
            encoder=self.encoder,
            decoder=self.decoder
        )

        # Define loss function and optimizer
        self.loss_function = VGAELoss(norm=2)
        self.optimizer = AdamW(params=self.model.parameters(), lr=config["LR"])

        self.train()

    def train_epoch(self):
        self.model.train()
        self.model.to(self.device)
        self.loss_function.to(self.device)

        preds = []
        targets = []
        total_loss = 0.

        try:
            adj = adj_matrix_from_edge_index(self.X, self.adj)
        except:
            adj=self.adj

        self.optimizer.zero_grad()
        adj_output, mu, logvar = self.model.to(self.device)(self.X,self.adj)

        loss = self.loss_function(adj_output, mu, logvar, adj.to(self.device))

        total_loss += loss.item()

        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()



    def train(self):
        for epoch in range(self.config["EPOCHS"]):
            self.train_epoch()

    def ComputeLikelihood(self):
        adj_output, _, _ = self.model(self.X,self.adj)
        adj_output = nn.Sigmoid()(adj_output)
        return adj_output.sum()

    def forward(self):
        return self.ComputeLikelihood()
