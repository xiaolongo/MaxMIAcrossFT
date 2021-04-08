import torch
import torch.nn as nn
from torch_geometric.nn import APPNP, GCNConv, SAGEConv

from utils import glorot, zeros


def corruption(x):
    return x[torch.randperm(x.size(0))]


def summary(z):
    return torch.sigmoid(z.mean(dim=0))


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.sigma = nn.PReLU(out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        z = self.sigma(self.conv(x, edge_index))
        return z


class SAGEEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = SAGEConv(in_dim, out_dim)
        self.sigma = nn.PReLU(out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        z = self.sigma(self.conv(x, edge_index))
        return z


class APPNPEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.prop = APPNP(K=10, alpha=0.2)
        self.lin = nn.Linear(in_dim, out_dim)
        self.sigma = nn.PReLU(out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = self.lin(x)
        z = self.sigma(self.prop(x, edge_index))
        return z


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
