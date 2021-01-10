import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from model.abstract import PJFModel
from model.layer import MLPLayers


class BERT(PJFModel):
    def __init__(self, config, pool):

        super(BERT, self).__init__(config, pool)

        self.emb_size = config['embedding_size']
        self.hd_size = config['hidden_size']
        self.dropout = config['dropout']

        # define layers and loss
        self.geek_emb = nn.Embedding.from_pretrained(
            torch.from_numpy(pool['geek_bert_vec']),
            freeze=True,
            padding_idx=0
        )
        self.geek_mlp = MLPLayers(
            [self.emb_size, self.hd_size],
            dropout=self.dropout,
            activation='tanh'
        )

        self.job_emb = nn.Embedding.from_pretrained(
            torch.from_numpy(pool['job_bert_vec']),
            freeze=True,
            padding_idx=0
        )
        self.job_mlp = MLPLayers(
            [self.emb_size, self.hd_size],
            dropout=self.dropout,
            activation='tanh'
        )

        self.pre_mlp = MLPLayers(
            [self.hd_size * 4, self.hd_size, 1],
            dropout=self.dropout,
            activation='tanh'
        )

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

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
        geek_vec = self.geek_mlp(geek_vec)
        job_vec = self.job_mlp(job_vec)
        score = self.pre_mlp(torch.cat([
            geek_vec, job_vec, geek_vec - job_vec, geek_vec * job_vec
        ], dim=1))
        return score.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction['label']
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
