import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from model.abstract import PJFModel
from model.layer import MLPLayers


class BERT(PJFModel):
    def __init__(self, config, pool):

        super(BERT, self).__init__(config, pool)

        self.embedding_size = config['embedding_size']
        self.hd_size = config['hidden_size']
        self.dropout = config['dropout']

        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size, self.hd_size),
            nn.BatchNorm1d(num_features=self.hd_size),
            nn.Sigmoid(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hd_size, 1)
        )

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, interaction):
        bert_vec = interaction['bert_vec']
        score = self.mlp(bert_vec).squeeze(-1)
        return score

    def calculate_loss(self, interaction):
        label = interaction['label']
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
