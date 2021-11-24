import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from model.abstract import PJFModel
from model.layer import BiBPRLoss, EmbLoss
from model.layer import GCNConv

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class LightGCNal(PJFModel):
    def __init__(self, config, pool):
        super(LightGCNal, self).__init__(config, pool)
        self.config = config
        
        # load dataset info
        self.interaction_matrix = pool.interaction_matrix.astype(np.float32)
        self.user_add_matrix = pool.user_add_matrix.astype(np.float32)
        self.job_add_matrix = pool.job_add_matrix.astype(np.float32)
        
        # load parameters info 
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.n_users = pool.geek_num
        self.n_items = pool.job_num

        # layers
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        self.gcn_conv = GCNConv(dim=self.latent_dim)
        self.mf_loss = BiBPRLoss()
        self.reg_loss = EmbLoss()
        self.loss = 0

        # generate intermediate data
        self.edge_index, self.edge_weight = self.get_norm_adj_mat()
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """

        # success edge
        row = torch.LongTensor(self.interaction_matrix.row)
        col = torch.LongTensor(self.interaction_matrix.col) + self.n_users
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index_suc = torch.cat([edge_index1, edge_index2], dim=1)

        # user_add edge
        row = torch.LongTensor(self.user_add_matrix.row)
        col = torch.LongTensor(self.user_add_matrix.col) + self.n_users
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index_user_add = torch.cat([edge_index1, edge_index2], dim=1)

        # job_add edge
        row = torch.LongTensor(self.job_add_matrix.row)
        col = torch.LongTensor(self.job_add_matrix.col) + self.n_users
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index_job_add = torch.cat([edge_index1, edge_index2], dim=1)

        # all edge
        edge_index = torch.cat([edge_index_suc, edge_index_user_add, edge_index_job_add], dim=1)
        # edge_index = edge_index_suc

        deg = degree(edge_index[0], self.n_users + self.n_items)
        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))

        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        user = interaction['geek_id']
        neg_user = interaction['neg_geek']
        pos_item = interaction['job_id']
        neg_item = interaction['neg_job']

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        neg_u_embeddings = user_all_embeddings[neg_user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BiBPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores_u = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        neg_scores_i = torch.mul(neg_u_embeddings, pos_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores_u, neg_scores_i)

        # calculate EMB Loss
        u_ego_embeddings = self.user_embedding(user)
        neg_u_ego_embeddings = self.user_embedding(neg_user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, neg_u_ego_embeddings)
        
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction['geek_id']
        item = interaction['job_id']

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores
        



