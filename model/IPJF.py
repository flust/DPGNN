import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn.init import normal_

from model.abstract import PJFModel
from model.layer import FusionLayer, HingeLoss


class IPJF(PJFModel):
    def __init__(self, config, pool):
        super(IPJF, self).__init__(config, pool)
        self.config = config
        self.pool = pool
        self.embedding_size = config['embedding_size']

        self.geek_fusion_layer = FusionLayer(self.embedding_size)
        self.job_fusion_layer = FusionLayer(self.embedding_size)

        self.w_job = nn.Linear(2 * self.embedding_size, 1)
        self.w_geek = nn.Linear(2 * self.embedding_size, 1)

        self.matching_mlp = nn.Sequential(
            nn.Linear(4 * self.embedding_size, 2 * self.embedding_size),
            nn.ReLU(),
            nn.Linear(2 * self.embedding_size, 1)
        )
        
        self.bert_lr = nn.Linear(config['BERT_embedding_size'],
                                self.embedding_size).to(self.config['device'])
        self._load_bert()
        self.sigmoid = nn.Sigmoid()
        self.loss = HingeLoss()
        
        self.apply(self._init_weights)

    def _load_bert(self):
        self.bert_user = torch.FloatTensor([]).to(self.config['device'])
        for i in range(self.pool.geek_num):
            geek_token = self.pool.geek_id2token[i]
            bert_id =  self.pool.geek_token2bertid[geek_token]
            bert_u_vec = self.pool.u_bert_vec[bert_id, :].unsqueeze(0).to(self.config['device'])
            self.bert_user = torch.cat([self.bert_user, bert_u_vec], dim=0)
            del bert_u_vec

        self.bert_job = torch.FloatTensor([]).to(self.config['device'])
        for i in range(self.pool.job_num):
            job_token = self.pool.job_id2token[i]
            bert_id =  self.pool.job_token2bertid[job_token]
            bert_j_vec = self.pool.j_bert_vec[bert_id].unsqueeze(0).to(self.config['device'])
            self.bert_job = torch.cat([self.bert_job, bert_j_vec], dim=0)
            del bert_j_vec

    def forward(self, geek_id, job_id):
        # bert
        geek_vec = self.bert_lr(self.bert_user[geek_id])
        job_vec = self.bert_lr(self.bert_job[job_id])

        f_s = self.geek_fusion_layer(geek_vec, job_vec)
        f_e = self.job_fusion_layer(geek_vec, job_vec)
        r_s = self.w_geek(f_s)  # [2048, 1]
        r_e = self.w_job(f_e)  # [2048, 1]

        r_m = self.matching_mlp(torch.cat((f_s, f_e), dim=1))
        return r_s, r_e, r_m

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        if isinstance(module, nn.Parameter):
            normal_(module.weight.data, 0, 0.01)

    def calculate_geek_loss(self, geek_sents, job_sents, neu_job, neg_job):
        r_s_pos, _ , r_m_pos = self.forward(geek_sents, job_sents)
        r_s, _ , r_m = self.forward(geek_sents, neu_job)
        r_s_neg, _ , r_m_neg = self.forward(geek_sents, neg_job)
        loss_s_i = self.loss(r_s_pos, r_s_neg) + self.loss(r_s, r_s_neg) # geek intention
        loss_s_m = self.loss(r_m_pos, r_m) + self.loss(r_m_pos, r_m_neg) # geek match
        return loss_s_i, loss_s_m

    def calculate_job_loss(self, geek_sents, job_sents, neu_geek, neg_geek):
        _, r_e_pos , r_m_pos = self.forward(geek_sents, job_sents)
        _, r_e , r_m = self.forward(neu_geek, job_sents)
        _, r_e_neg , r_m_neg = self.forward(neg_geek, job_sents)
        loss_e_i = self.loss(r_e_pos, r_e_neg) + self.loss(r_e, r_e_neg) # job intention
        loss_e_m = self.loss(r_m_pos, r_m) + self.loss(r_m_pos, r_m_neg) # job match
        return loss_e_i, loss_e_m

    def calculate_loss(self, interaction):
        geek_sents = interaction['geek_id']
        job_sents = interaction['job_id']
        neu_job = interaction['neu_job_id'] # 中性岗位
        neg_job = interaction['neg_job_id'] # 负岗位
        neu_geek = interaction['neu_geek_id'] # 中性用户
        neg_geek = interaction['neg_geek_id'] # 负用户

        loss_s_i, loss_s_m = self.calculate_geek_loss(geek_sents, job_sents, neu_job, neg_job)
        loss_e_i, loss_e_m = self.calculate_job_loss(geek_sents, job_sents, neu_geek, neg_geek)

        loss = loss_s_i + loss_s_m + loss_e_i + loss_e_m
        return loss

    def predict(self, interaction):
        geek_id = interaction['geek_id']
        job_id = interaction['job_id']
        _, _, match_score = self.forward(geek_id, job_id)
        return torch.sigmoid(match_score)
