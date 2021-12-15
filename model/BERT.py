import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from model.abstract import PJFModel
from model.layer import MLPLayers, FusionLayer


class BERT(PJFModel):
    def __init__(self, config, pool):
        super(BERT, self).__init__(config, pool)
        self.embedding_size = config['embedding_size']
        self.hd_size = config['hidden_size']
        self.dropout = config['dropout']
        self.config = config
        self.pool = pool
        self.n_users = pool.geek_num
        self.n_items = pool.job_num
        self._load_bert()
        
        self.user_mlp = nn.Linear(self.embedding_size, self.hd_size)
        self.job_mlp = nn.Linear(self.embedding_size, self.hd_size)

        self.fusion_layer = FusionLayer(self.hd_size)
        self.fc = nn.Linear(2 * self.hd_size, 1)
        
        self.loss = nn.BCEWithLogitsLoss()

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

    def forward(self, interaction):
        geek_id = interaction['geek_id']
        job_id = interaction['job_id']
        geek_vec = self.user_mlp(self.bert_user[geek_id])
        job_vec = self.job_mlp(self.bert_job[job_id])

        score = self.fusion_layer(geek_vec, job_vec)
        score = self.fc(score)
        return score.squeeze()

    def forward_neg(self, interaction):
        geek_id = interaction['geek_id']
        job_id = interaction['neg_job']
        geek_vec = self.user_mlp(self.bert_user[geek_id])
        job_vec = self.job_mlp(self.bert_job[job_id])

        score = self.fusion_layer(geek_vec, job_vec)
        score = self.fc(score)
        return score.squeeze()

    def calculate_loss(self, interaction):
        output_pos = self.forward(interaction)
        output_neg = self.forward_neg(interaction)
        label_pos = interaction['label_pos'].to(self.config['device']).squeeze()
        label_neg = interaction['label_neg'].to(self.config['device']).squeeze()
        
        loss = self.loss(output_pos, label_pos)
        loss += self.loss(output_neg, label_neg)
        return loss

    def predict(self, interaction):
        return self.forward(interaction)
