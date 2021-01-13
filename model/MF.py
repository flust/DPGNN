import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from model.abstract import PJFModel


class MF(PJFModel):
    def __init__(self, config, pool):

        super(MF, self).__init__(config, pool)

        self.embedding_size = config['embedding_size']

        # define layers and loss
        self.geek_emb = nn.Embedding(self.geek_num, self.embedding_size)
        self.job_emb = nn.Embedding(self.job_num, self.embedding_size)
        self.geek_b = nn.Embedding(self.geek_num, 1)
        self.job_b = nn.Embedding(self.job_num, 1)
        self.miu = nn.Parameter(torch.rand(1, ), requires_grad=True)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([config['pos_weight']]))

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
        score = torch.sum(torch.mul(geek_vec, job_vec), dim=1) \
            + self.geek_b(geek_id).squeeze() \
            + self.job_b(job_id).squeeze() \
            + self.miu
        return score

    def calculate_loss(self, interaction):
        label = interaction['label']
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
