import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from model.abstract import PJFModel
from model.layer import BiBPRLoss, EmbLoss
from model.layer import GCNConv

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class MultiGCNsl(PJFModel):
    def __init__(self, config, pool):
        super(MultiGCNsl, self).__init__(config, pool)
        self.config = config
        
        # load dataset info
        self.interaction_matrix = pool.interaction_matrix.astype(np.float32)
        self.user_add_matrix = pool.user_add_matrix.astype(np.float32)
        self.job_add_matrix = pool.job_add_matrix.astype(np.float32)
        
        # load parameters info 
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.omega = config['omega']
        self.n_users = pool.geek_num
        self.n_items = pool.job_num

        # layers
        self.user_embedding_a = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding_a = nn.Embedding(self.n_items, self.latent_dim)
        self.user_embedding_p = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding_p = nn.Embedding(self.n_items, self.latent_dim)
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
        #   user a node: [0 ~~~ n_users] (len: n_users)
        #   item p node: [n_users + 1 ~~~ n_users + 1 + n_items] (len: n_users)
        #   user p node: [~] (len: n_users)
        #   item a node: [~] (len: n_items)
        n_all = self.n_users + self.n_items

        # success edge
        row = torch.LongTensor(self.interaction_matrix.row)
        col = torch.LongTensor(self.interaction_matrix.col) + self.n_users
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index3 = torch.stack([row + n_all, col + n_all])
        edge_index4 = torch.stack([col + n_all, row + n_all])
        edge_index_suc = torch.cat([edge_index1, edge_index2, edge_index3, edge_index4], dim=1)

        # user_add edge
        row = torch.LongTensor(self.user_add_matrix.row)
        col = torch.LongTensor(self.user_add_matrix.col) + self.n_users
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index_user_add = torch.cat([edge_index1, edge_index2], dim=1)

        # job_add edge
        row = torch.LongTensor(self.job_add_matrix.row)
        col = torch.LongTensor(self.job_add_matrix.col) + self.n_users
        edge_index1 = torch.stack([row + n_all, col + n_all])
        edge_index2 = torch.stack([col + n_all, row + n_all])
        edge_index_job_add = torch.cat([edge_index1, edge_index2], dim=1)

        # self edge
        geek = torch.LongTensor(torch.arange(0, self.n_users))
        job = torch.LongTensor(torch.arange(0, self.n_items) + self.n_users)
        edge_index_geek_1 = torch.stack([geek, geek + n_all])
        edge_index_geek_2 = torch.stack([geek + n_all, geek])
        edge_index_job_1 = torch.stack([job, job + n_all])
        edge_index_job_2 = torch.stack([job + n_all, job])
        edge_index_self = torch.cat([edge_index_geek_1, edge_index_geek_2, edge_index_job_1, edge_index_job_2], dim=1)

        # all edge
        edge_index = torch.cat([edge_index_suc, edge_index_user_add, edge_index_job_add, edge_index_self], dim=1)

        deg = degree(edge_index[0], (self.n_users + self.n_items) * 2)
        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))

        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings_a = self.user_embedding_a.weight
        item_embeddings_a = self.item_embedding_a.weight
        user_embeddings_p = self.user_embedding_p.weight
        item_embeddings_p = self.item_embedding_p.weight
        ego_embeddings = torch.cat([user_embeddings_a,
                            item_embeddings_p,
                            user_embeddings_p,
                            item_embeddings_a], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_e_a, item_e_p, user_e_p, item_e_a = torch.split(lightgcn_all_embeddings, 
                    [self.n_users, self.n_items, self.n_users, self.n_items])
        return user_e_a, item_e_p, user_e_p, item_e_a

    def calculate_loss(self, interaction):
        user = interaction['geek_id']
        pos_item = interaction['job_id']
        neg_user = interaction['neg_geek']
        neg_item = interaction['neg_job']

        user_e_a, item_e_p, user_e_p, item_e_a = self.forward()

        # user active
        u_e_a = user_e_a[user]
        n_u_e_a = user_e_a[neg_user]
        # item negative
        i_e_p = item_e_p[pos_item]
        n_i_e_p = item_e_p[neg_item]

        # user negative
        u_e_p = user_e_p[user]
        n_u_e_p = user_e_p[neg_user]
        # item active
        i_e_a = item_e_a[pos_item]
        n_i_e_a = item_e_a[neg_item]

        I_geek = torch.mul(u_e_a, i_e_p).sum(dim=1)
        I_job = torch.mul(u_e_p, i_e_a).sum(dim=1)

        n_I_geek_1 = torch.mul(u_e_a, n_i_e_p).sum(dim=1)
        n_I_job_1 = torch.mul(u_e_p, n_i_e_a).sum(dim=1)

        n_I_geek_2 = torch.mul(n_u_e_a, i_e_p).sum(dim=1)
        n_I_job_2 = torch.mul(n_u_e_p, i_e_a).sum(dim=1)

        # calculate BPR Loss
        pos_scores = I_geek + I_job
        neg_scores_u = n_I_geek_1 + n_I_job_1
        neg_scores_i = n_I_geek_2 + n_I_job_2

        mf_loss_u = self.mf_loss(pos_scores, neg_scores_u, neg_scores_i, self.omega)

        # calculate Emb Loss
        u_ego_embeddings_a = self.user_embedding_a(user)
        u_ego_embeddings_p = self.user_embedding_p(user)
        pos_ego_embeddings_a = self.item_embedding_a(pos_item)
        pos_ego_embeddings_p = self.item_embedding_p(pos_item)
        neg_ego_embeddings_a = self.item_embedding_a(neg_item)
        neg_ego_embeddings_p = self.item_embedding_p(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings_a, u_ego_embeddings_p, 
                                    pos_ego_embeddings_a, pos_ego_embeddings_p,
                                    neg_ego_embeddings_a, neg_ego_embeddings_p)
        
        loss = mf_loss_u + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction['geek_id']
        item = interaction['job_id']

        user_e_a, item_e_p, user_e_p, item_e_a = self.forward()

        # user activate
        u_e_a = user_e_a[user]
        # item negative
        i_e_p = item_e_p[item]

        # user negative
        u_e_p = user_e_p[user]
        # item negative
        i_e_a = item_e_a[item]

        I_geek = torch.mul(u_e_a, i_e_p).sum(dim=1)
        I_job = torch.mul(u_e_p, i_e_a).sum(dim=1)
        # calculate BPR Loss
        scores = I_geek + I_job
        return scores
        



