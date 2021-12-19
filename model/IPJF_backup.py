import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn.init import normal_

from model.abstract import PJFModel
from model.layer import FusionLayer

class HingeLoss(torch.nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.delta = 0.05

    def forward(self, pos_score, neg_score):
        hinge_loss = torch.clamp(pos_score - neg_score + self.delta, min=0).mean()
        return hinge_loss

# temporal narrow convolution
# in: geek_sents
#       sentences_1, sentences_2, ... , sentences_p
#       {r_1, r_2, ..., r_k}, {r_1', r_2', ...r_k'}, ..., {}
# out: geek_sents_emb
#       J_s \in R^{p*d}
#       
class TextCNN(nn.Module):
    def __init__(self, channels, kernel_size, pool_size, dim, method='max'):
        super(TextCNN, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size[0]),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size[1]),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, dim))
        )
        if method is 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, dim))
        elif method is 'mean':
            self.pool = nn.AdaptiveAvgPool2d((1, dim))
        else:
            raise ValueError('method {} not exist'.format(method))

    def forward(self, x):
        x = self.net1(x)  # [2048, 20, 13, 64]
        x = self.net2(x).squeeze(2) # [2048, 20, 64]
        x = self.pool(x).squeeze(1) # [2048, 64]
        return x

# co-attention
# in: geek_sents_emb / job_sents_emb
#       J_s / R_s
# out: geek_emb / job_emb
#       J / R

class IPJF(PJFModel):
    def __init__(self, config, pool):
        super(IPJF, self).__init__(config, pool)
        self.embedding_size = config['embedding_size']
        self.geek_channels = config['geek_max_sent_num'] #15
        self.job_channels = config['job_max_sent_num']   #20

        self.emb = nn.Embedding(pool.wd_num, self.embedding_size, padding_idx=0)

        self.geek_layer = TextCNN(
            channels=self.geek_channels,
            kernel_size=[(5, 1), (3, 1)],
            pool_size=(2, 1),
            dim=self.embedding_size,
            method='max'
        )

        self.job_layer = TextCNN(
            channels=self.job_channels,
            kernel_size=[(5, 1), (5, 1)],
            pool_size=(2, 1),
            dim=self.embedding_size,
            method='mean'
        )

        # self.w_att = nn.Parameter(torch.rand(self.embedding_size, self.embedding_size))
        # self.register_parameter('w_att', self.w_att)

        # self.job_attn = nn.Linear(self.embedding_size, 1)
        # self.geek_attn = nn.Linear(self.embedding_size, 1)

        self.geek_fusion_layer = FusionLayer(self.embedding_size)
        self.job_fusion_layer = FusionLayer(self.embedding_size)

        self.w_job = nn.Linear(2 * self.embedding_size, 1)
        self.w_geek = nn.Linear(2 * self.embedding_size, 1)

        self.matching_mlp = nn.Sequential(
            nn.Linear(4 * self.embedding_size, 2 * self.embedding_size),
            nn.BatchNorm1d(num_features=2 * self.embedding_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2 * self.embedding_size, 1)
        )

        self.BERT_e_size = config['BERT_output_size']
        self.bert_lr = nn.Linear(config['BERT_embedding_size'],
                                self.BERT_e_size).to(self.config['device'])
        self._load_bert()
        
        self.sigmoid = nn.Sigmoid()
        self.loss = HingeLoss()
        self.apply(self._init_weights)

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

    def forward(self, geek_id, job_id):
        # import pdb
        # pdb.set_trace()
        # geek_vec = self.emb(geek_sents)
        # job_vec = self.emb(job_sents)
        
        # geek_vec = self.geek_layer(geek_vec) # [2048, 64]
        # job_vec = self.job_layer(job_vec)  # [2048, 64]

        f_s = self.geek_fusion_layer(geek_vec, job_vec)
        f_e = self.job_fusion_layer(geek_vec, job_vec)
        r_s = self.w_geek(f_s)  # [2048, 1]
        r_e = self.w_job(f_e)  # [2048, 1]

        r_m = self.matching_mlp(torch.cat((f_s, f_e), dim=1))
        return r_s, r_e, r_m

    # def forward(self, interaction):
    #     geek_sents = interaction['geek_sents']
    #     job_sents = interaction['job_sents']
    #     f_s, f_e, r_s, r_e = self.calculate_f_s(geek_sents, job_sents)

    #     return x.squeeze(1)

        # geek_matrix = self.geek_layer(geek_vec) # [2048, 15, 64]
        # job_matrix = self.job_layer(job_vec)  # [2048, 20, 64]
        # self.w_att [64, 64]

        # # IPJF attention
        # A = job_matrix \cdot self.w_att \cdot geek_matrix
        # A = torch.matmul(geek_matrix, self.w_att)
        # A = torch.matmul(A, job_matrix.permute(0, 2, 1))  # [2048, 15, 20]
        # 
        # geek_attn = A.sum(dim = 2)  # [2048, 15]
        # geek_attn = torch.softmax(geek_attn, dim=1) # [2048, 15]
        # geek_vec = torch.sum((geek_matrix.permute(2, 0, 1) * geek_attn).permute(1, 2, 0),
        #                         dim=1)  # [2048, 64]

        # job_attn = A.sum(dim = 1)
        # job_attn = torch.softmax(job_attn, dim=1)
        # job_vec = torch.sum((job_matrix.permute(2, 0, 1) * job_attn).permute(1, 2, 0),
        #                         dim=1)   # [2048, 64]

        # attention
        # geek_attn = self.geek_attn(geek_matrix)
        # geek_attn = torch.softmax(geek_attn, dim=1)
        # geek_vec = torch.sum(geek_matrix * geek_attn, dim=1)

        # job_attn = self.geek_attn(job_matrix)
        # job_attn = torch.softmax(job_attn, dim=1)
        # job_vec = torch.sum(job_matrix * job_attn, dim=1)

        # import pdb
        # pdb.set_trace()
        # fusion layer
        # geek_f = self.geek_fusion_layer(geek_vec, job_vec)
        # job_f = self.job_fusion_layer(geek_vec, job_vec)

        # geek_r = self.w_geek(geek_f)  # [2048, 1]
        # job_r = self.w_job(job_f)  # [2048, 1]
        # # =====
        # # import pdb
        # # pdb.set_trace()
        
        # # x = geek_vec * job_vec

        # # x = self.mlp(x).squeeze(1)
        # # return x    
        # # =====
        # x = self.matching_mlp(torch.cat((geek_f, job_f), 1))
        # return x.squeeze(1)

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

    def calculate_geek_loss(self, geek_sents, job_sents, neu_geek, neg_geek):
        _, r_e_pos , r_m_pos = self.forward(geek_sents, job_sents)
        _, r_e , r_m = self.forward(neu_geek, job_sents)
        _, r_e_neg , r_m_neg = self.forward(neg_geek, job_sents)
        loss_e_i = self.loss(r_e_pos, r_e_neg) + self.loss(r_e, r_e_neg) # job intention
        loss_e_m = self.loss(r_m_pos, r_m) + self.loss(r_m_pos, r_m_neg) # job match
        return loss_e_i, loss_e_m

    def calculate_loss(self, interaction):
        geek_sents = interaction['geek_sents']
        job_sents = interaction['job_sents']
        neu_job = interaction['neu_job_sents'] # 中性岗位
        neg_job = interaction['neg_job_sents'] # 负岗位
        neu_geek = interaction['neu_geek_sents'] # 中性用户
        neg_geek = interaction['neg_geek_sents'] # 负用户

        loss_s_i, loss_s_m = self.calculate_geek_loss(geek_sents, job_sents, neu_job, neg_job)
        loss_e_i, loss_e_m = self.calculate_geek_loss(geek_sents, job_sents, neu_geek, neg_geek)

        loss = loss_s_i + loss_s_m + loss_e_i + loss_e_m
        return loss

    def predict(self, interaction):
        geek_sents = interaction['geek_sents']
        job_sents = interaction['job_sents']
        _, _, match_score = self.forward(geek_sents, job_sents)
        return torch.sigmoid(match_score)