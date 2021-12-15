import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from model.abstract import PJFModel
from model.layer import BPRLoss, EmbLoss
from model.layer import GCNConv

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import pdb


class MultiGCNBERT(PJFModel):
    def __init__(self, config, pool):
        super(MultiGCNBERT, self).__init__(config, pool)
        self.config = config
        self.pool = pool
        
        # load dataset info
        self.interaction_matrix = pool.interaction_matrix.astype(np.float32)
        self.user_add_matrix = pool.user_add_matrix.astype(np.float32)
        self.job_add_matrix = pool.job_add_matrix.astype(np.float32)
        
        # load parameters info 
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.mul_weight = config['mutual_weight']
        self.temperature = config['temperature']
        self.n_users = pool.geek_num
        self.n_items = pool.job_num

        # layers
        self.user_embedding_a = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding_a = nn.Embedding(self.n_items, self.latent_dim)
        self.user_embedding_p = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding_p = nn.Embedding(self.n_items, self.latent_dim)
        self.gcn_conv = GCNConv(dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.mutual_loss = nn.CrossEntropyLoss().to(self.device)
        self.loss = 0

        # bert part
        self.ADD_BERT = config['ADD_BERT']
        self.BERT_e_size = 0
        if self.ADD_BERT:
            self.BERT_e_size = config['BERT_output_size']
            self.bert_lr = nn.Linear(config['BERT_embedding_size'],
                                        self.BERT_e_size).to(self.config['device'])
            self._load_bert()

        # generate intermediate data
        self.edge_index, self.edge_weight = self.get_norm_adj_mat()
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def _load_bert(self):
        self.bert_user = torch.FloatTensor([]).to(self.config['device'])
        for i in range(self.n_users):
            geek_token = self.pool.geek_id2token[i]
            bert_id =  self.pool.geek_token2bertid[geek_token]
            bert_u_vec = self.pool.u_bert_vec[bert_id, :].unsqueeze(0).to(self.config['device'])
            self.bert_user = torch.cat([self.bert_user, bert_u_vec], dim=0)
            del bert_u_vec

        self.bert_job = torch.FloatTensor([]).to(self.config['device'])
        for i in range(self.n_items):
            job_token = self.pool.job_id2token[i]
            bert_id =  self.pool.job_token2bertid[job_token]
            bert_j_vec = self.pool.j_bert_vec[bert_id].unsqueeze(0).to(self.config['device'])
            self.bert_job = torch.cat([self.bert_job, bert_j_vec], dim=0)
            del bert_j_vec

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

        if self.ADD_BERT:
            self.bert_u = self.bert_lr(self.bert_user)
            self.bert_j = self.bert_lr(self.bert_job)

            bert_e = torch.cat([self.bert_u,
                                self.bert_j, 
                                self.bert_u, 
                                self.bert_j], dim = 0)
            return torch.cat([ego_embeddings, bert_e], dim=1)

        return ego_embeddings

    def info_nce_loss(self, index, is_user):
        # pdb.set_trace
        all_embeddings = self.get_ego_embeddings()
        user_e_a, item_e_p, user_e_p, item_e_a = torch.split(all_embeddings, 
                    [self.n_users, self.n_items, self.n_users, self.n_items])
        if is_user:
            u_e_a = F.normalize(user_e_a[index], dim=1)
            u_e_p = F.normalize(user_e_p[index], dim=1)
            similarity_matrix = torch.matmul(u_e_a, u_e_p.T)
        else:
            i_e_a = F.normalize(item_e_a[index], dim=1)
            i_e_p = F.normalize(item_e_p[index], dim=1)
            similarity_matrix = torch.matmul(i_e_a, i_e_p.T)

        mask = torch.eye(index.shape[0], dtype=torch.bool).to(self.device)

        positives = similarity_matrix[mask].view(index.shape[0], -1)
        negatives = similarity_matrix[~mask].view(index.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        logits = logits / self.temperature

        return logits, labels

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
        item = interaction['job_id']
        neg_user = interaction['neg_geek']
        neg_item = interaction['neg_job']

        user_e_a, item_e_p, user_e_p, item_e_a = self.forward()

        # user active
        u_e_a = user_e_a[user]
        n_u_e_a = user_e_a[neg_user]
        # item negative
        i_e_p = item_e_p[item]
        n_i_e_p = item_e_p[neg_item]

        # user negative
        u_e_p = user_e_p[user]
        n_u_e_p = user_e_p[neg_user]
        # item active
        i_e_a = item_e_a[item]
        n_i_e_a = item_e_a[neg_item]

        r_pos = torch.mul(u_e_a, i_e_p).sum(dim=1)
        s_pos = torch.mul(u_e_p, i_e_a).sum(dim=1)

        r_neg1 = torch.mul(u_e_a, n_i_e_p).sum(dim=1)
        s_neg1 = torch.mul(u_e_p, n_i_e_a).sum(dim=1)

        r_neg2 = torch.mul(n_u_e_a, i_e_p).sum(dim=1)
        s_neg2 = torch.mul(n_u_e_p, i_e_a).sum(dim=1)

        # calculate BPR Loss
        # pos_scores = I_geek + I_job
        # neg_scores_u = n_I_geek_1 + n_I_job_1
        # neg_scores_i = n_I_geek_2 + n_I_job_2

        mf_loss_u = self.mf_loss(2 * r_pos + 2 * s_pos, r_neg1 + s_neg1 + r_neg2 + s_neg2)

        # calculate Emb Loss
        u_ego_embeddings_a = self.user_embedding_a(user)
        u_ego_embeddings_p = self.user_embedding_p(user)
        pos_ego_embeddings_a = self.item_embedding_a(item)
        pos_ego_embeddings_p = self.item_embedding_p(item)
        neg_ego_embeddings_a = self.item_embedding_a(neg_item)
        neg_ego_embeddings_p = self.item_embedding_p(neg_item)
        neg_u_ego_embeddings_a = self.user_embedding_a(neg_user)
        neg_u_ego_embeddings_p = self.user_embedding_p(neg_user)

        reg_loss = self.reg_loss(u_ego_embeddings_a, u_ego_embeddings_p, 
                                    pos_ego_embeddings_a, pos_ego_embeddings_p,
                                    neg_ego_embeddings_a, neg_ego_embeddings_p,
                                    neg_u_ego_embeddings_a, neg_u_ego_embeddings_p)
        
        loss = mf_loss_u + self.reg_weight * reg_loss

        logits_user, labels = self.info_nce_loss(user, is_user=True)
        loss += self.mul_weight * self.mutual_loss(logits_user, labels)

        logits_job, labels = self.info_nce_loss(item, is_user=False)
        loss += self.mul_weight * self.mutual_loss(logits_job, labels)

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
        



