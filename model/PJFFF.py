import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from model.abstract import PJFModel
from model.layer import BPRLoss, EmbLoss
from model.layer import GCNConv



class PJFFF(PJFModel):
    def __init__(self, config, pool):
        super(PJFFF, self).__init__(config, pool)
        self.config = config
        self.pool = pool
        
        # load parameters info 
        self.n_users = pool.geek_num
        self.n_items = pool.job_num       

        # layers
        self.embedding_size = config['BERT_output_size']
        self.hd_size = self.embedding_size
        self.bert_lr = nn.Linear(config['BERT_embedding_size'],
                                    self.embedding_size).to(self.config['device'])
        self._load_bert()

        self.job_biLSTM = nn.LSTM(
            input_size=2 * self.embedding_size,
            hidden_size=self.hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.geek_biLSTM = nn.LSTM(
            input_size=2 * self.embedding_size,
            hidden_size=self.hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.job_layer = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.geek_layer = nn.Linear(self.embedding_size * 2, self.embedding_size)

        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
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

    def get_fg_E(self, interaction, neg=False):
        geek_id = interaction['geek_id']
        if neg:
            job_id = interaction['neg_job']
        else:
            job_id = interaction['job_id']
        f_e = self.bert_lr(self.bert_user[geek_id])
        g_e = self.bert_lr(self.bert_job[job_id])
        return f_e, g_e

    def forward_E(self, interaction, neg=False):
        f_e, g_e = self.get_fg_E(interaction, neg)
        score_E = torch.mul(f_e, g_e).sum(dim=1)
        return score_E

    def forward_I(self, interaction):
        f_e, g_e = self.get_fg_E(interaction)
        his_job = interaction['his_job'].long()
        # his_job_len = interaction['his_job_len']
        his_geek = interaction['his_geek'].long()
        # his_geek_len = interaction['his_geek_len']
        neg_his_geek = interaction['neg_his_geek'].long()
        # neg_his_geek_len = interaction['neg_his_geek_len']

        his_f_e = self.bert_lr(self.bert_user[his_geek]) # [2048, 100, 32]
        his_g_e = self.bert_lr(self.bert_job[his_job]) # [2048, 100, 32]
        neg_his_f_e = self.bert_lr(self.bert_user[neg_his_geek])

        g_i = torch.cat((his_f_e, g_e.unsqueeze(1).repeat(1, his_f_e.shape[1], 1)), dim=2) # [2048, 100, 64]
        f_i = torch.cat((his_g_e, f_e.unsqueeze(1).repeat(1, his_g_e.shape[1], 1)), dim=2)
        neg_g_i = torch.cat((neg_his_f_e, g_e.unsqueeze(1).repeat(1, neg_his_f_e.shape[1], 1)), dim=2)

        g_i, _ = self.job_biLSTM(g_i)
        g_i = torch.sum(g_i, dim=1)
        f_i, _ = self.geek_biLSTM(f_i)
        f_i = torch.sum(f_i, dim=1)
        neg_g_i, _ = self.job_biLSTM(neg_g_i)
        neg_g_i = torch.sum(neg_g_i, dim=1)

        g_i = self.job_layer(g_i)
        f_i = self.geek_layer(f_i)
        neg_g_i = self.job_layer(neg_g_i)

        return f_i, g_i, neg_g_i

    def calculate_loss(self, interaction):
        score_E = self.forward_E(interaction)
        score_E_neg = self.forward_E(interaction, neg=True)
        f_i, g_i, neg_g_i = self.forward_I(interaction)
        score_I = torch.mul(f_i, g_i).sum(dim=1)
        score_I_neg = torch.mul(f_i, neg_g_i).sum(dim=1)

        label_pos = interaction['label_pos'].to(self.config['device']).squeeze()
        label_neg = interaction['label_neg'].to(self.config['device']).squeeze()

        loss_E = self.loss(score_E, label_pos) + self.loss(score_E_neg, label_neg)
        loss_I = self.loss(score_I, label_pos) + self.loss(score_I_neg, label_neg)
        return loss_E + loss_I

    def predict(self, interaction):
        score_E = self.forward_E(interaction)
        score_E = self.sigmoid(score_E)
        f_i, g_i, _ = self.forward_I(interaction)
        score_I = torch.mul(f_i, g_i).sum(dim=1)
        return score_E + score_I
        



