import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from model.abstract import PJFModel


class MultiMF(PJFModel):
    def __init__(self, config, pool, dataset):

        super(MultiMF, self).__init__(config, pool)

        self.embedding_size = config['embedding_size']

        # define layers and loss
        self.geek_p = nn.Embedding(self.geek_num, self.embedding_size)
        self.geek_c = nn.Embedding(self.geek_num, self.embedding_size)
        self.job_p = nn.Embedding(self.job_num, self.embedding_size)
        self.job_c = nn.Embedding(self.job_num, self.embedding_size)
        self.geek_p_b = nn.Embedding(self.geek_num, 1)
        self.geek_c_b = nn.Embedding(self.geek_num, 1)
        self.job_p_b = nn.Embedding(self.job_num, 1)
        self.job_c_b = nn.Embedding(self.job_num, 1)
        self.miu_g = nn.Parameter(torch.rand(1, ), requires_grad=True)
        self.miu_j = nn.Parameter(torch.rand(1, ), requires_grad=True)

        self.sigmoid = nn.Sigmoid()

        # self.loss = nn.BCEWithLogitsLoss()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([config['pos_weight']]))

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, interaction):
        geek_id = interaction['geek_id']
        job_id = interaction['job_id']
        geek_p = self.geek_p(geek_id)
        geek_c = self.geek_c(geek_id)
        job_p = self.job_p(job_id)
        job_c = self.job_c(job_id)

        geek_p_score = torch.sum(torch.mul(geek_p, job_c), dim=1) \
                + self.geek_p_b(geek_id).squeeze() \
                + self.job_c_b(job_id).squeeze() \
                + self.miu_g
        job_p_score = torch.sum(torch.mul(geek_c, job_p), dim=1) \
                + self.geek_c_b(geek_id).squeeze() \
                + self.job_p_b(job_id).squeeze() \
                + self.miu_j
        return geek_p_score, job_p_score

    def calculate_loss(self, interaction):
        label = interaction['label']
        # import pdb
        # pdb.set_trace()
        # zero = torch.zeros_like(label)
        # one = torch.ones_like(label) * 0.1
        # geek_label = torch.where(label == 2, one, label)
        # geek_label = torch.where(label == 3, zero, geek_label)

        # job_label = torch.where(label == 3, one, label)
        # job_label = torch.where(label == 2, zero, job_label)

        # geek_p_score, job_p_score = self.forward(interaction)
        # return self.loss(geek_p_score, geek_label) + self.loss(job_p_score, job_label)

        geek_p_score, job_p_score = self.forward(interaction)
        # return self.loss(geek_p_score, label) + self.loss(job_p_score, label)
        return self.loss(geek_p_score + job_p_score, label)


    def predict(self, interaction):
        geek_p_score, job_p_score = self.forward(interaction)
        return self.sigmoid(geek_p_score + job_p_score)



# import torch
# import torch.nn as nn
# from torch.nn.init import xavier_normal_

# from model.abstract import PJFModel
# from model.layer import MLPLayers


# class MultiMF(nn.Module):
#     def __init__(self, config, pool, MF_geek, MF_job):
#         super(MultiMF, self).__init__()
#         self.MF_geek = MF_geek
#         self.MF_job = MF_job
#         self.embedding_size = config['embedding_size']
#         self.hd_size = config['hidden_size']
#         self.dropout = config['dropout']
#         self.mlp = nn.Sequential(
#             nn.Linear(self.embedding_size * 2, self.hd_size),
#             nn.Dropout(p=self.dropout),
#             nn.Linear(self.hd_size, 1)
#         )
#         self.sigmoid = nn.Sigmoid()
#         self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([config['pos_weight']]))

#     def _init_weights(self, module):
#         if isinstance(module, nn.Embedding):
#             xavier_normal_(module.weight.data)

#     def forward(self, interaction):
#         geek_id = interaction['geek_id']
#         job_id = interaction['job_id']
        
#         geek_vec_1 = self.MF_geek.geek_emb(geek_id)
#         job_vec_1 = self.MF_geek.job_emb(job_id)
#         vec_1 = torch.mul(geek_vec_1, job_vec_1)

#         geek_vec_2 = self.MF_job.geek_emb(geek_id)
#         job_vec_2 = self.MF_job.job_emb(job_id)
#         vec_2 = torch.mul(geek_vec_2, job_vec_2)

#         # import pdb
#         # pdb.set_trace()
#         score = self.mlp(torch.cat((vec_1, vec_2), 1)).squeeze()
#         # print(score.shape)
#         score += self.MF_job.geek_b(geek_id).squeeze() \
#             + self.MF_job.job_b(job_id).squeeze() \
#             + self.MF_job.miu
#         score += self.MF_geek.geek_b(geek_id).squeeze() \
#             + self.MF_geek.job_b(job_id).squeeze() \
#             + self.MF_geek.miu
#         return score.squeeze()

#     def calculate_loss(self, interaction):
#         label = interaction['label']
#         output = self.forward(interaction)
#         return self.loss(output, label)

#     def predict(self, interaction):
#         # return (self.MF_geek.predict(interaction) + self.MF_job.predict(interaction)) / 2
#         return self.sigmoid(self.forward(interaction))

