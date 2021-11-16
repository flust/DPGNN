import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from model.abstract import PJFModel
from model.layer import BPRLoss

class MF(PJFModel):
    def __init__(self, config, pool):

        super(MF, self).__init__(config, pool)
        self.config = config
        self.pool = pool
        self.embedding_size = config['embedding_size']

        # define layers and loss
        self.geek_emb = nn.Embedding(self.geek_num, self.embedding_size)
        self.job_emb = nn.Embedding(self.job_num, self.embedding_size)
        # self.geek_b = nn.Embedding(self.geek_num, 1)
        # self.job_b = nn.Embedding(self.job_num, 1)
        # self.miu = nn.Parameter(torch.rand(1, ), requires_grad=True)

        # bert part
        # self.ADD_BERT = config['ADD_BERT']
        # self.BERT_e_size = 0
        # if self.ADD_BERT:
        #     self.BERT_e_size = config['BERT_output_size']
        #     self.bert_lr = nn.Linear(config['BERT_embedding_size'],
        #                 self.BERT_e_size).to(self.config['device'])
        #     self._load_bert()

        # self.sigmoid = nn.Sigmoid()
        # self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([config['pos_weight']]))
        self.loss = BPRLoss().to(config['device'])

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, interaction):
        geek_id = interaction['geek_id']
        job_id = interaction['job_id']

        geek_vec = self.geek_emb(geek_id)
        job_vec = self.job_emb(job_id)

        # if self.ADD_BERT:
        #     self.bert_u = self.bert_lr(self.bert_user)
        #     self.bert_j = self.bert_lr(self.bert_job)
        #     geek_vec = torch.cat([self.geek_emb(geek_id), self.bert_u[geek_id]], dim=1)
        #     job_vec = torch.cat([self.job_emb(job_id), self.bert_j[job_id]], dim=1)
            
        score = torch.sum(torch.mul(geek_vec, job_vec), dim=1) \
            # + self.geek_b(geek_id).squeeze() \
            # + self.job_b(job_id).squeeze() \
            # + self.miu

        return score

    def calculate_loss(self, interaction):
        # label = interaction['label']
        # geek_id = interaction['geek_id']
        # job_id = interaction['job_id']
        # scores = self.forward(interaction)
        # return self.loss(scores, label)

        geek_id = interaction['geek_id']
        job_id = interaction['job_id']
        neg_id = interaction['neg_job']

        geek_vec = self.geek_emb(geek_id)
        job_vec = self.job_emb(job_id)
        neg_vec = self.job_emb(neg_id)

        pos_scores = torch.mul(geek_vec, job_vec).sum(dim=1)
        neg_scores = torch.mul(geek_vec, neg_vec).sum(dim=1)

        return self.loss(pos_scores, neg_scores)

    def predict(self, interaction):
        return self.forward(interaction)

    # def _load_bert(self):
        # self.bert_user = torch.FloatTensor([]).to(self.config['device'])
        # for i in range(self.geek_num):
        #     geek_token = self.pool.geek_id2token[i]
        #     bert_id =  self.pool.geek_token2bertid[geek_token]
        #     bert_u_vec = self.pool.u_bert_vec[bert_id, :].unsqueeze(0).to(self.config['device'])
        #     self.bert_user = torch.cat([self.bert_user, bert_u_vec], dim=0)

        # self.bert_job = torch.FloatTensor([]).to(self.config['device'])
        # for i in range(self.job_num):
        #     job_token = self.pool.job_id2token[i]
        #     bert_id =  self.pool.job_token2bertid[job_token]
        #     bert_j_vec = self.pool.j_bert_vec[bert_id].unsqueeze(0).to(self.config['device'])
        #     self.bert_job = torch.cat([self.bert_job, bert_j_vec], dim=0)