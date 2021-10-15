import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn.init import normal_
from model.abstract import PJFModel
from model.layer import GCNConv, GATConv
import pdb

class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss
        

class BGPJF(PJFModel):
    def __init__(self, config, pool):
        super(BGPJF, self).__init__(config, pool)
        self.config = config
        self.pool = pool
        self.n_users = pool.geek_num
        self.n_items = pool.job_num

        self.success_edge = pool.success_edge
        self.user_add_edge = pool.user_add_edge
        self.job_add_edge = pool.job_add_edge
        
        # load parameters info 
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN

        # create edges
        self.edge_index = self.create_edge().to(config['device'])
        # self.edge_matrix = [] * 2 * (self.n_users + self.n_items)

        # layers
        self.user_embedding_c = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding_c = nn.Embedding(self.n_items, self.latent_dim)
        self.user_embedding_p = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding_p = nn.Embedding(self.n_items, self.latent_dim)

        # bias
        self.geek_b = nn.Embedding(self.geek_num, 1)
        self.job_b = nn.Embedding(self.job_num, 1)
        self.miu = nn.Parameter(torch.rand(1, ), requires_grad=True)

        # gcn layers
        gcn_modules = []
        for i in range(self.n_layers):
            # gcn_modules.append(GCNConv(self.latent_dim, self.latent_dim))
            gcn_modules.append(GATConv(self.latent_dim, self.latent_dim))
        self.gcn_layers = nn.Sequential(*gcn_modules)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([config['pos_weight']]))
        # self.loss_fct = nn.BCELoss()
        self.loss_fct = BPRLoss()

        self.apply(self._init_weights)

    def get_edge(self, edges, u_p=True, j_p=True):
        geek_id_p = edges[0].unsqueeze(0)
        job_id_c = edges[1].unsqueeze(0) + self.n_users
        geek_id_c = edges[0].unsqueeze(0) + self.n_users + self.n_items
        job_id_p = edges[1].unsqueeze(0) + self.n_users + self.n_items + self.n_users
        
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
        # user_success_edge = self.get_edge(self.dataset['train_g'], u_p=True, j_p=True)

        # In job success data, geek_c <-> job_p  &&  geek_p <-> job_c
        # job_success_edge = self.get_edge(self.dataset['train_j'], u_p=True, j_p=True)

        # success edge, geek_c <-> job_p  &&  geek_p <-> job_c
        success_edge = self.get_edge(self.success_edge, u_p=True, j_p=True)

        # In geek addfriend data, geek_p <-> job_c
        user_addfriend_edge = self.get_edge(self.user_add_edge, u_p=True, j_p=False)

        # In job addfriend data, geek_c <-> job_p
        job_addfriend_edge = self.get_edge(self.job_add_edge, u_p=False, j_p=True)
                               
        # geek_p <-> geek_c  &&  job_p <-> job_c
        # self_edge = self.get_self_edge()

        # combine all edges
        # edges = torch.cat((user_success_edge, 
        #                     job_success_edge, 
        #                     user_addfriend_edge,
        #                     job_addfriend_edge,
        #                     self_edge), 1)
        edges = torch.cat((success_edge,
                            user_addfriend_edge,
                            job_addfriend_edge), 1)
        # make edges bidirected
        # edges = torch.cat((edges, edges[[1,0]]), 1)
        return edges

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        if isinstance(module, nn.Linear):
            # if self.init_method == 'norm':
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings_c = self.user_embedding_c.weight
        item_embeddings_c = self.item_embedding_c.weight
        user_embeddings_p = self.user_embedding_p.weight
        item_embeddings_p = self.item_embedding_p.weight

        return torch.cat([user_embeddings_p,
                            item_embeddings_c,
                            user_embeddings_c,
                            item_embeddings_p], dim=0)

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for i in range(self.n_layers):
            all_embeddings = self.gcn_layers[i](all_embeddings, self.edge_index)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_e_p, item_e_c, user_e_c, item_e_p = torch.split(lightgcn_all_embeddings, 
                    [self.n_users, self.n_items, self.n_users, self.n_items])
        return user_e_p, item_e_c, user_e_c, item_e_p

    def calculate_score(self, user, item):
        r"""calculate score for user and item
        """
        user_e_p, item_e_c, user_e_c, item_e_p = self.forward()

        u_e_c = user_e_c[user]
        i_e_c = item_e_c[item]
        u_e_p = user_e_p[user]
        i_e_p = item_e_p[item]
        I_geek = torch.mul(u_e_p, i_e_c).sum(dim=1)
        I_job = torch.mul(u_e_c, i_e_p).sum(dim=1)
        scores = I_geek + I_job \
            + self.geek_b(user).squeeze() \
            + self.job_b(item).squeeze() \
            + self.miu
        return scores

    def predict(self, interaction):
        user = interaction['geek_id']
        item = interaction['job_id']
        scores = self.calculate_score(user, item)
        return self.sigmoid(scores)

    def get_perm(self, x):
        return list(torch.randperm(max(self.pool.sample_n, x))[:self.pool.sample_n])

    def mutual_loss(self, interaction):
        user_e_p, item_e_c, user_e_c, item_e_p = self.forward()
        pos_geek = interaction['geek_id']
        pos_job = interaction['job_id']

        user_self_score = torch.mul(user_e_p[pos_geek], user_e_c[pos_geek]).sum(dim=1)
        user_self_score = (user_self_score.repeat(self.pool.sample_n, 1).T).reshape(-1)

        item_self_score = torch.mul(item_e_p[pos_job], item_e_c[pos_job]).sum(dim=1)
        item_self_score = (item_self_score.repeat(self.pool.sample_n, 1).T).reshape(-1)

        # geek neg sample
        g_n_idx = interaction['geek_neg_sample']
        geek_neg_list = torch.gather(self.pool.geek2jobs_neg[pos_geek].to(self.config['device']), 1, g_n_idx)
        u_pos = (pos_geek.repeat(self.pool.sample_n, 1).T).reshape(-1)
        j_neg = geek_neg_list.view(-1).type(torch.long)
        geek_neg_score = torch.mul(user_e_p[u_pos] + user_e_c[u_pos], item_e_p[j_neg] + item_e_c[j_neg]).sum(dim=1)

        # job neg sample 
        j_n_idx = interaction['job_neg_sample']
        job_neg_list = torch.gather(self.pool.job2geeks_neg[pos_job].to(self.config['device']), 1, j_n_idx)
        u_neg = job_neg_list.view(-1).type(torch.long)
        j_pos = (pos_job.repeat(self.pool.sample_n, 1).T).reshape(-1)
        geek_neg_score = torch.mul(user_e_p[u_neg] + user_e_c[u_neg], item_e_p[j_pos] + item_e_c[j_pos]).sum(dim=1)

        loss = self.loss_fct(user_self_score, geek_neg_score) + self.loss_fct(item_self_score, geek_neg_score)
        return loss

        # pdb.set_trace()

        # user_e_p, item_e_c, user_e_c, item_e_p = self.forward()
        # user_self_score = torch.mul(user_e_p, user_e_c).sum(dim=1)
        # item_self_score = torch.mul(item_e_p, item_e_c).sum(dim=1)
        
        # user_list = list(torch.arange(self.n_users))
        # item_list = list(torch.arange(self.n_items))

        # user_neg_list = torch.Tensor(list(map(self.get_perm, user_list)))
        # item_neg_list = torch.Tensor(list(map(self.get_perm, item_list)))
        
        # # geek neg score
        # u_self = ((user_e_p + user_e_c).repeat(self.pool.sample_n, 1).T).reshape(-1)
        # user_neg_list = user_neg_list.view(-1).type(torch.long)

        # u_neight = u_self * (item_e_c[user_neg_list] + item_e_p[user_neg_list])
        # # geek_neg_score = geek_neg_score.reshape(pos_geek.shape[0], -1).mean(dim = 1)
        # loss = self.loss_fct(u_self, u_neight) + self.loss_fct(j_self, j_neight)

        # u_n_s = max(self.sample_n, self.pool.job2geeks_neg_num[job_id]) # 
        # self.job_neg_sample = torch.randperm(j_n_s)[:self.sample_n]

        # j_n_s = max(self.sample_n, self.pool.job2geeks_neg_num[job_id]) # 
        # self.job_neg_sample = torch.randperm(j_n_s)[:self.sample_n]

        # neg_score = torch.mul((user_e_p + user_e_c) * (item_e_p + item_e_c)).sum(dim=1)

        # return user_pos_score + item_pos_score

    def bilateral_loss(self, scores, interaction):
        pos_idx = interaction['label'] == 1
        pos_geek = interaction['geek_id'][pos_idx] # torch.Size([674])
        pos_job = interaction['job_id'][pos_idx] # torch.Size([674])
        pos_score = scores[pos_idx] # torch.Size([674])

        # geek neg sample
        g_n_idx = interaction['geek_neg_sample'][pos_idx]
        geek_neg_list = torch.gather(self.pool.geek2jobs_neg[pos_geek].to(self.config['device']), 1, g_n_idx)

        # job neg sample 
        j_n_idx = interaction['job_neg_sample'][pos_idx]
        job_neg_list = torch.gather(self.pool.job2geeks_neg[pos_job].to(self.config['device']), 1, j_n_idx)

        # geek neg score
        u_pos = (pos_geek.repeat(self.pool.sample_n, 1).T).reshape(-1)
        j_neg = geek_neg_list.view(-1).type(torch.long)
        geek_neg_score = self.calculate_score(u_pos, j_neg)
        geek_neg_score = geek_neg_score.reshape(pos_geek.shape[0], -1).mean(dim = 1)

        # job neg score
        u_neg = job_neg_list.view(-1).type(torch.long)
        j_pos = (pos_job.repeat(self.pool.sample_n, 1).T).reshape(-1)
        job_neg_score = self.calculate_score(u_neg, j_pos)
        job_neg_score = job_neg_score.reshape(pos_geek.shape[0], -1).mean(dim = 1)

        loss = self.loss_fct(pos_score, geek_neg_score) + self.loss_fct(pos_score, geek_neg_score)
        return loss

    def calculate_loss(self, interaction):
        # calculate BPR Loss
        user = interaction['geek_id']
        item = interaction['job_id']
        scores = self.calculate_score(user, item)

        label = interaction['label']  # 4096

        scores = self.sigmoid(scores)

        main_loss = self.loss(scores, label)
        lambda_1 = self.config['lambda_1']
        bilateral_loss = self.bilateral_loss(scores, interaction)
        lambda_2 = self.config['lambda_2']
        mutual_loss = self.mutual_loss(interaction)

        # scores = self.sigmoid(scores)
        # return main_loss + lambda_1 * bilateral_loss
        return main_loss + lambda_1 * bilateral_loss + lambda_2 * mutual_loss



