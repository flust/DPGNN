import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_

from model.abstract import PJFModel
from model.layer import MLPLayers


class BPJFNN(PJFModel):
    def __init__(self, config, pool):

        super(BPJFNN, self).__init__(config, pool)

        self.embedding_size = config['embedding_size']
        self.hd_size = config['hidden_size']
        self.dropout = config['dropout']
        self.num_layers = config['num_layers']

        # define layers and loss
        self.emb = nn.Embedding(pool.wd_num, self.embedding_size, padding_idx=0)
        self.job_biLSTM = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.geek_biLSTM = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.mlp = MLPLayers(
            layers=[self.hd_size * 3 * 2, 1],
            dropout=self.dropout
        )

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.LSTM):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, interaction):
        geek_longsent = interaction['geek_longsent']
        job_longsent = interaction['job_longsent']
        geek_vec, job_vec = self.emb(geek_longsent), self.emb(job_longsent)
        geek_vec, _ = self.geek_biLSTM(geek_vec)
        job_vec, _ = self.job_biLSTM(job_vec)
        geek_vec = torch.mean(geek_vec, dim=1)
        job_vec = torch.mean(job_vec, dim=1)
        x = torch.cat([job_vec, geek_vec, job_vec - geek_vec], dim=1)
        x = self.mlp(x).squeeze(1)
        return x

    def calculate_loss(self, interaction):
        label = interaction['label']
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
