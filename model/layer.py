import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch.nn.init import normal_
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, remove_self_loops, softmax

from typing import Union, Tuple, Optional, Any


class MLPLayers(nn.Module):
    """ MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples:

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(self, layers, dropout=0., activation='relu', bn=False, init_method=None):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == 'norm':
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def activation_layer(activation_name='relu', emb_dim=None):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_name.lower() == 'tanh':
            activation = nn.Tanh()
        elif activation_name.lower() == 'relu':
            activation = nn.ReLU()
        elif activation_name.lower() == 'leakyrelu':
            activation = nn.LeakyReLU()
        elif activation_name.lower() == 'none':
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError("activation function {} is not implemented".format(activation_name))

    return activation


class SimpleFusionLayer(nn.Module):
    def __init__(self, hd_size):
        super(SimpleFusionLayer, self).__init__()
        self.fc = nn.Linear(hd_size * 4, hd_size)

    def forward(self, a, b):
        assert a.shape == b.shape
        x = torch.cat([a, b, a * b, a - b], dim=-1)
        x = self.fc(x)
        x = torch.tanh(x)
        return x


class FusionLayer(nn.Module):
    def __init__(self, hd_size):
        super(FusionLayer, self).__init__()
        self.m = SimpleFusionLayer(hd_size)
        self.g = nn.Sequential(
            nn.Linear(hd_size * 2, 1),
            nn.Sigmoid()
        )

    def _single_layer(self, a, b):
        ma = self.m(a, b)
        x = torch.cat([a, b], dim=-1)
        ga = self.g(x)
        return ga * ma + (1 - ga) * a

    def forward(self, a, b):
        assert a.shape == b.shape
        a = self._single_layer(a, b)
        b = self._single_layer(b, a)
        return torch.cat([a, b], dim=-1)


class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class BiBPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BiBPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score_g, neg_score_j, omega = 1):
        loss = - torch.log(self.gamma + torch.sigmoid(pos_score - neg_score_g)).mean() \
                - omega * torch.log(self.gamma + torch.sigmoid(pos_score - neg_score_j)).mean()
        return loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings
    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


class GCNConv(MessagePassing):
    def __init__(self, dim):
        super(GCNConv, self).__init__(aggr='add')
        self.dim = dim

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)


# MultiGNN v1
# class GNNConv(MessagePassing):
#     def __init__(self, dim):
#         super(GNNConv, self).__init__(aggr='add')
#         self.dim = dim

#     def forward(self, x, edge_index, edge_weight):
#         return self.propagate(edge_index, x=x, edge_weight=edge_weight)

#     def message(self, x_j, edge_weight):
#         return edge_weight.view(-1, 1) * x_j

#     def __repr__(self):
#         return '{}({})'.format(self.__class__.__name__, self.dim)


class GNNConv(MessagePassing):
    def __init__(self, dim):
        super(GNNConv, self).__init__(node_dim=0, aggr='add')
        self.dim = dim
        self.att_src = Parameter(torch.Tensor(1, dim))
        self.att_dst = Parameter(torch.Tensor(1, dim))

    def forward(self, x, edge_index, edge_weight):
        # x: torch.Size([209448, 128])
        # edge_index: torch.Size([2, 5706724])
        # edge_weight: torch.Size([5706724])
        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)
        # import pdb
        # pdb.set_trace()


        return self.propagate(edge_index, x=x, alpha=alpha, edge_weight=edge_weight)

    def message(self, x_j, alpha_j, edge_weight, index):
        # x_j: torch.Size([5706724, 128])
        # edge_weight: torch.Size([5706724])
        # index: torch.Size([5706724])
        
        alpha = F.leaky_relu(alpha_j)
        alpha = softmax(alpha, index)
        # alpha: torch.Size([5706724])
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)
