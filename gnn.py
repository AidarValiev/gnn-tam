from dataclasses import dataclass

import torch
import torch.nn as nn

from common_types import TAMHyperParameters
from gsl import GSL


class GCLayer(nn.Module):
    """
    Graph convolution layer.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)

    def forward(self, adj, X):
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        h = self.dense(X)
        norm = adj.sum(1)**(-1/2)
        h = norm[None, :] * adj * norm[:, None] @ h
        return h



class GNN_TAM(nn.Module):
    """
    Model architecture from the paper "Graph Neural Networks with Trainable
    Adjacency Matrices for Fault Diagnosis on Multivariate Sensor Data".
    https://doi.org/10.1109/ACCESS.2024.3481331
    """
    def __init__(
        self,
        hyper_params: TAMHyperParameters,
    ):
        super(GNN_TAM, self).__init__()
        self.window_size = hyper_params.window_size
        self.nhidden = hyper_params.n_hidden
        self.idx = nn.Buffer(torch.arange(hyper_params.n_nodes), persistent=False)
        self.adj = [0 for i in range(hyper_params.n_gnn)]
        self.h = [0 for i in range(hyper_params.n_gnn)]
        self.skip = [0 for i in range(hyper_params.n_gnn)]
        self.z = nn.Buffer(torch.ones(hyper_params.n_nodes, hyper_params.n_nodes) - torch.eye(hyper_params.n_nodes), persistent=False)
        self.n_gnn = hyper_params.n_gnn

        self.gsl = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.bnorm1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.bnorm2 = nn.ModuleList()

        for i in range(self.n_gnn):
            self.gsl.append(GSL(hyper_params.gsl_type, hyper_params.n_nodes,
                                hyper_params.window_size, hyper_params.alpha, hyper_params.k))
            self.conv1.append(GCLayer(hyper_params.window_size, hyper_params.n_hidden))
            self.bnorm1.append(nn.BatchNorm1d(hyper_params.n_nodes))
            self.conv2.append(GCLayer(hyper_params.n_hidden, hyper_params.n_hidden))
            self.bnorm2.append(nn.BatchNorm1d(hyper_params.n_nodes))

        self.fc = nn.Linear(hyper_params.n_gnn*hyper_params.n_hidden, hyper_params.n_classes)

    def forward(self, X):
        for i in range(self.n_gnn):
            self.adj[i] = self.gsl[i](self.idx)
            self.adj[i] = self.adj[i] * self.z
            self.h[i] = self.conv1[i](self.adj[i], X).relu()
            self.h[i] = self.bnorm1[i](self.h[i])
            self.skip[i], _ = torch.min(self.h[i], dim=1)
            self.h[i] = self.conv2[i](self.adj[i], self.h[i]).relu()
            self.h[i] = self.bnorm2[i](self.h[i])
            self.h[i], _ = torch.min(self.h[i], dim=1)
            self.h[i] = self.h[i] + self.skip[i]

        h = torch.cat(self.h, 1)
        output = self.fc(h)

        return output

    def get_adj(self):
        return self.adj
