import numpy as np
import torch
import torch.nn as nn

from torch.nn.init import xavier_normal_
from model.abstract import PJFModel
from model.layer import GCNConv


class LightGCN(PJFModel):
    def __init__(self, config, pool, dataset):
        super(LightGCN, self).__init__(config, pool)
        self.config = config
        self.dataset = dataset
        self.n_users = pool.geek_num
        self.n_items = pool.job_num
        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN

        # create edges
        self.edge_index = self.create_edge().to(self.config['device'])

        # layers
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        # self.geek_b = nn.Embedding(self.geek_num, 1)
        # self.job_b = nn.Embedding(self.job_num, 1)
        # self.miu = nn.Parameter(torch.rand(1, ), requires_grad=True)

        # add gcn layers
        self.gcn_layers = []
        for i in range(self.n_layers):
            self.gcn_layers.append(GCNConv(self.latent_dim, self.latent_dim))

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([config['pos_weight']]))
        self.apply(self._init_weights)

    def create_edge(self):
        # create edges: use data which label is 1 
        self.geek_id = self.dataset.geek_ids[self.dataset.labels == 1].unsqueeze(0)
        self.job_id = self.dataset.job_ids[self.dataset.labels == 1].unsqueeze(0)
        # create edges: use data that exist
        # self.geek_id = self.dataset.geek_ids.unsqueeze(0)
        # self.job_id = self.dataset.job_ids.unsqueeze(0)

        # edges_0: geek_node -> job_node
        edges_0 = torch.cat((self.geek_id, self.n_users + self.job_id), 0)
        # edges_1: job_node -> geek_node
        edges_1 = torch.cat((self.n_users + self.job_id, self.geek_id), 0)
        # edges: concat(edges_0, edges_1)
        edges = torch.cat((edges_0, edges_1), 1)
        # edges = torch.cat((edges, edges[[1,0]]), 1)
        return edges

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

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
        # follow LightGCN in recbole
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for i in range(self.n_layers):
            all_embeddings = self.gcn_layers[i](all_embeddings, self.edge_index)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_score(self, interaction):
        r"""calculate score for user and item

        Returns:
            torch.mul(user_embedding, item_embedding)
        """
        user = interaction['geek_id']
        item = interaction['job_id']
        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1) \
                    # + self.geek_b(user).squeeze() \
                    # + self.job_b(item).squeeze() \
                    # + self.miu
        return scores

    def predict(self, interaction):
        scores = self.calculate_score(interaction)
        return self.sigmoid(scores)

    def calculate_loss(self, interaction):
        # calculate BPR Loss
        scores = self.calculate_score(interaction)
        label = interaction['label']

        return self.loss(scores, label)