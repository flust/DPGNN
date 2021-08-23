import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from torch.nn.init import xavier_normal_
from model.abstract import PJFModel
from model.layer import GCNConv
from scipy.sparse import coo_matrix
from model.layer import FusionLayer


class MultiPJF(PJFModel):
    def __init__(self, config, pool, dataset):
        super(MultiPJF, self).__init__(config, pool)
        self.config = config
        self.dataset = dataset
        self.n_users = pool.geek_num
        self.n_items = pool.job_num
        self.pool = pool
        self.config = config
        # load parameters info 
        self.embedding_size = config['embedding_size']
        self.GCN_e_size = self.embedding_size
        self.BERT_e_size = config['BERT_output_size']
        # self.GCN_e_size = config['GCN_embedding_size']  # int type:the embedding size of lightGCN
        self.GCN_n_layers = config['GCN_layers']  # int type:the layer num of lightGCN

        self.ADD_BERT = config['ADD_BERT']
        self.ADD_STAR = config['ADD_STAR']

        # create edges
        self.edge_index = self.create_edge().to(config['device'])

        # layers
        self.user_embedding_c = nn.Embedding(self.n_users, self.GCN_e_size)
        self.item_embedding_c = nn.Embedding(self.n_items, self.GCN_e_size)
        self.user_embedding_p = nn.Embedding(self.n_users, self.GCN_e_size)
        self.item_embedding_p = nn.Embedding(self.n_items, self.GCN_e_size)

        # gcn layers
        gcn_modules = []
        for i in range(self.GCN_n_layers):
            gcn_modules.append(GCNConv(self.GCN_e_size, self.GCN_e_size))
        self.gcn_layers = nn.Sequential(*gcn_modules)
        # star_in_features = 2 * self.GCN_e_size
        final_in_features = 6 * self.GCN_e_size

        # bert part
        if self.ADD_BERT:
            self.bert_lr = nn.Linear(config['BERT_embedding_size'],
                        self.BERT_e_size).to(self.config['device'])
            self._load_bert()
            # star_in_features = 2 * (self.GCN_e_size + self.BERT_e_size)
            final_in_features = 6 * (self.GCN_e_size + self.BERT_e_size)

        # star part
        if self.ADD_STAR:
            # self.job_lr = nn.Linear(star_in_features, 1)
            # self.geek_lr = nn.Linear(star_in_features, 1)
            self.final_lr = nn.Linear(final_in_features, 1)

        # bias
        self.geek_b = nn.Embedding(self.geek_num, 1)
        self.job_b = nn.Embedding(self.job_num, 1)
        self.miu = nn.Parameter(torch.rand(1, ), requires_grad=True)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([config['pos_weight']]))
        self.apply(self._init_weights)

    def get_edge(self, data, u_p=True, j_p=True):
        mask = (data.labels >= 1)
        geek_id_p = data.geek_ids[mask].unsqueeze(0)
        job_id_c = data.job_ids[mask].unsqueeze(0) + self.n_users
        geek_id_c = data.geek_ids[mask].unsqueeze(0) + self.n_users + self.n_items
        job_id_p = data.job_ids[mask].unsqueeze(0) + self.n_users + self.n_items + self.n_users
        
        u_p_edge = torch.cat((geek_id_p, job_id_c), 0)
        j_p_edge = torch.cat((job_id_p, geek_id_c), 0)
        if u_p and j_p:
            return torch.cat((u_p_edge, j_p_edge), 1)
        elif u_p:
            return u_p_edge
        elif j_p:
            return j_p_edge
        else:
            raise ValueError("At least one of u_p and j_p must be true")

    def get_self_edge(self):
        # geek_p <-> geek_c
        geek_begin = torch.arange(0, self.n_users).unsqueeze(0).long()
        geek_end = geek_begin + self.n_users + self.n_items
        geek_edge = torch.cat((geek_begin, geek_end), 0)

        # job_p <-> job_c
        job_begin = torch.arange(self.n_users, self.n_users + self.n_items).unsqueeze(0).long()
        job_end = job_begin + self.n_users + self.n_items
        job_edge = torch.cat((job_begin, job_end), 0)
        return torch.cat((geek_edge, job_edge), 1)

    def create_edge(self):
        # create edges
        # node id: In the following order
        #   user p node: [0 ~~~ n_users] (len: n_users)
        #   item c node: [n_users + 1 ~~~ n_users + 1 + n_items] (len: n_users)
        #   user c node: [~] (len: n_users)
        #   item p node: [~] (len: n_items)

        # In geek success data, geek_p <-> job_c  &&  geek_c <-> job_p
        user_success_edge = self.get_edge(self.dataset['train_g'], u_p=True, j_p=True)

        # In job success data, geek_c <-> job_p  &&  geek_p <-> job_c
        job_success_edge = self.get_edge(self.dataset['train_j'], u_p=True, j_p=True)
        
        # In geek addfriend data, geek_p <-> job_c
        user_addfriend_edge = self.get_edge(self.dataset['add_user'], u_p=True, j_p=False)

        # In job addfriend data, geek_c <-> job_p
        job_addfriend_edge = self.get_edge(self.dataset['add_job'], u_p=False, j_p=True)
                               
        # geek_p <-> geek_c  &&  job_p <-> job_c
        self_edge = self.get_self_edge()

        # combine all edges
        # edges = torch.cat((user_success_edge, 
        #                     job_success_edge, 
        #                     user_addfriend_edge,
        #                     job_addfriend_edge,
        #                     self_edge), 1)
        edges = torch.cat((
                            user_success_edge, 
                            job_success_edge, 
                            user_addfriend_edge,
                            job_addfriend_edge,
                            self_edge), 1)
        # make edges bidirected
        # edges = torch.cat((edges, edges[[1,0]]), 1)
        return edges

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

        self.bert_job = torch.FloatTensor([]).to(self.config['device'])
        for i in range(self.n_items):
            job_token = self.pool.job_id2token[i]
            bert_id =  self.pool.job_token2bertid[job_token]
            bert_j_vec = self.pool.j_bert_vec[bert_id].unsqueeze(0).to(self.config['device'])
            self.bert_job = torch.cat([self.bert_job, bert_j_vec], dim=0)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings_c = self.user_embedding_c.weight
        item_embeddings_c = self.item_embedding_c.weight
        user_embeddings_p = self.user_embedding_p.weight
        item_embeddings_p = self.item_embedding_p.weight
        id_e = torch.cat([user_embeddings_p,
                            item_embeddings_c,
                            user_embeddings_c,
                            item_embeddings_p], dim=0)
        if not self.ADD_BERT:
            return id_e
        else:
            self.bert_u = self.bert_lr(self.bert_user)
            self.bert_j = self.bert_lr(self.bert_job)

            bert_e = torch.cat([self.bert_u,
                                self.bert_j, 
                                self.bert_u, 
                                self.bert_j], dim = 0)
            return torch.cat([id_e, bert_e], dim=1)

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for i in range(self.GCN_n_layers):
            all_embeddings = self.gcn_layers[i](all_embeddings, self.edge_index)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_e_p, item_e_c, user_e_c, item_e_p = torch.split(lightgcn_all_embeddings, 
                    [self.n_users, self.n_items, self.n_users, self.n_items])
        return user_e_p, item_e_c, user_e_c, item_e_p

    def get_star(self, id, direction, e_c, e_p):
        # id  tensor  shape:4096
        if direction == 0:
            id2ids = self.pool.geek2jobs
        else:
            id2ids = self.pool.job2geeks

        from operator import itemgetter
        import random
        id = id.cpu().numpy().tolist()
        # import pdb
        # pdb.set_trace()
        ids = list(itemgetter(*id)(id2ids))

        # ids = torch.Tensor([id2ids[id.cpu().numpy().tolist()]]).squeeze().type(torch.long)

        p_star = torch.FloatTensor().to(self.config['device'])
        c_star = torch.FloatTensor().to(self.config['device'])
        for i in ids:
            i = random.sample(i, min(len(i),10))
            p = e_p[i].mean(dim=0) if e_p[i].dim() == 2 else e_p[i]
            c = e_c[i].mean(dim=0) if e_c[i].dim() == 2 else e_c[i]
            p_star = torch.cat((p_star, p.unsqueeze(0)), dim=0)
            c_star = torch.cat((c_star, c.unsqueeze(0)), dim=0)

        return p_star, c_star

    def calculate_score(self, interaction):
        r"""calculate score for user and item

        Returns:
            torch.mul(user_embedding, item_embedding)
        """
        user = interaction['geek_id']   # shape 4096
        item = interaction['job_id']
        user_e_p, item_e_c, user_e_c, item_e_p = self.forward()

        u_e_c = user_e_c[user]
        i_e_c = item_e_c[item]
        u_e_p = user_e_p[user]
        i_e_p = item_e_p[item]

        if not self.ADD_STAR:
            scores = torch.mul(u_e_p, i_e_c).sum(dim=1) \
                + torch.mul(u_e_c, i_e_p).sum(dim=1) \
                + self.geek_b(user).squeeze() \
                + self.job_b(item).squeeze() \
                + self.miu
        else:
            I_geek = torch.mul(u_e_p, i_e_c)
            I_job = torch.mul(u_e_c, i_e_p)

            # job_p_star = torch.FloatTensor().to(self.config['device'])
            # job_c_star = torch.FloatTensor().to(self.config['device'])
            # geek_p_star = torch.FloatTensor().to(self.config['device'])
            # geek_c_star = torch.FloatTensor().to(self.config['device'])
            # for u in user:
            #     jp, jc = self.get_star(u, 0, item_e_c, item_e_p)
            #     job_p_star = torch.cat((job_p_star, jp.unsqueeze(0)), dim=0)
            #     job_c_star = torch.cat((job_c_star, jc.unsqueeze(0)), dim=0)

            # for i in item:
            #     gp, gc = self.get_star(i, 1, user_e_c, user_e_p)
            #     geek_p_star = torch.cat((geek_p_star, gp.unsqueeze(0)), dim=0)
            #     geek_c_star = torch.cat((geek_c_star, gc.unsqueeze(0)), dim=0)

            job_p_star, job_c_star = self.get_star(user, 0, item_e_c, item_e_p)
            geek_p_star, geek_c_star = self.get_star(item, 1, user_e_c, user_e_p)
            
            scores = self.final_lr(torch.cat([job_p_star,
                                                job_c_star,
                                                geek_p_star,
                                                geek_c_star,
                                                I_geek,
                                                I_job], dim=1))
        return scores.squeeze()

    def predict(self, interaction):
        scores = self.calculate_score(interaction)
        # return self.sigmoid(scores)
        return scores

    def calculate_loss(self, interaction):
        # calculate BPR Loss
        scores = self.calculate_score(interaction)
        label = interaction['label']
        # scores = self.sigmoid(scores)
        return self.loss(scores, label)

