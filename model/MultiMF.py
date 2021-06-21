import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from model.abstract import PJFModel
from model.layer import MLPLayers


class MultiMF(nn.Module):
    def __init__(self, config, pool, MF_geek, MF_job):
        super(MultiMF, self).__init__()
        self.MF_geek = MF_geek
        self.MF_job = MF_job
        self.embedding_size = config['embedding_size']
        self.hd_size = config['hidden_size']
        self.dropout = config['dropout']
        # print(type(self.embedding_size))
        # print(self.embedding_size)
        # print(type(self.hd_size))
        # print(self.hd_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size * 2, self.hd_size),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hd_size, 1)
        )
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([config['pos_weight']]))

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, interaction):
        geek_id = interaction['geek_id']
        job_id = interaction['job_id']
        
        geek_vec_1 = self.MF_geek.geek_emb(geek_id)
        job_vec_1 = self.MF_geek.job_emb(job_id)
        vec_1 = torch.mul(geek_vec_1, job_vec_1)

        geek_vec_2 = self.MF_job.geek_emb(geek_id)
        job_vec_2 = self.MF_job.job_emb(job_id)
        vec_2 = torch.mul(geek_vec_2, job_vec_2)

        # import pdb
        # pdb.set_trace()
        score = self.mlp(torch.cat((vec_1, vec_2), 1)).squeeze()
        # print(score.shape)
        score += self.MF_job.geek_b(geek_id).squeeze() \
            + self.MF_job.job_b(job_id).squeeze() \
            + self.MF_job.miu
        score += self.MF_geek.geek_b(geek_id).squeeze() \
            + self.MF_geek.job_b(job_id).squeeze() \
            + self.MF_geek.miu
        return score.squeeze()



        # user = interaction[self.USER_ID]
        # item = interaction[self.ITEM_ID]

        # user_all_embeddings, item_all_embeddings = self.MF_geek.forward()

        # u_embeddings = user_all_embeddings[user]
        # i_embeddings = item_all_embeddings[item]
        # scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)

    def calculate_loss(self, interaction):
        label = interaction['label']
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        # return (self.MF_geek.predict(interaction) + self.MF_job.predict(interaction)) / 2
        return self.sigmoid(self.forward(interaction))

# class MultiMF(nn.Module):
    # def __init__(self, MF_geek, MF_job):
    #     super(MultiMF, self).__init__(config, pool)

    #     self.embedding_size = config['embedding_size']

    #     # define layers and loss
    #     self.geek_emb_1 = nn.Embedding(self.geek_num, self.embedding_size)
    #     self.job_emb_1 = nn.Embedding(self.job_num, self.embedding_size)

    #     self.geek_emb_2 = nn.Embedding(self.geek_num, self.embedding_size)
    #     self.job_emb_2 = nn.Embedding(self.job_num, self.embedding_size)

    #     self.linear = nn.Linear(self.embedding_size * 2, 1)

    # #     self.geek_b_1 = nn.Embedding(self.geek_num, 1)
    #     self.job_b_1 = nn.Embedding(self.job_num, 1)
    #     self.miu_1 = nn.Parameter(torch.rand(1, ), requires_grad=True)

    #     self.sigmoid = nn.Sigmoid()
    #     self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([config['pos_weight']]))

    #     # parameters initialization
    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Embedding):
    #         xavier_normal_(module.weight.data)

    # def forward(self, interaction):
    #     geek_id = interaction['geek_id']
    #     job_id = interaction['job_id']
    #     # direction = interaction['direction']
    #     geek_vec_1 = self.geek_emb_1(geek_id)
    #     job_vec_1 = self.job_emb_1(job_id)
    #     geek_vec_2 = self.geek_emb_2(geek_id)
    #     job_vec_2 = self.job_emb_2(job_id)
    #     score = torch.sum(torch.mul(geek_vec_1, job_vec_1), dim=1) \
    #         + torch.sum(torch.mul(geek_vec_2, job_vec_2), dim=1) \
    #         + self.geek_b_1(geek_id).squeeze() \
    #         + self.job_b_1(job_id).squeeze() \
    #         + self.miu_1
    #     return score

    # def calculate_loss(self, interaction):
    #     label = interaction['label']
    #     output = self.forward(interaction)
    #     return self.loss(output, label)

    # def predict(self, interaction):
    #     return self.sigmoid(self.forward(interaction))
