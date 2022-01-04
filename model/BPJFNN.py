import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_

from model.abstract import PJFModel
from model.layer import MLPLayers


class BPJFNN(PJFModel):
    def __init__(self, config, pool):

        super(BPJFNN, self).__init__(config, pool)
        self.config = config
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
            layers=[self.hd_size * 3 * 2, self.hd_size, 1],
            dropout=self.dropout,
            activation='tanh'
        )

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

    def _single_bpj_layer(self, interaction, token, pre=''):
        longsent = interaction[f'{pre}{token}_longsent']
        longsent_len = interaction[f'{pre}{token}_longsent_len']
        vec = self.emb(longsent)
        vec, _ = getattr(self, f'{token}_biLSTM')(vec)
        vec = torch.sum(vec, dim=1) / longsent_len.unsqueeze(-1)
        return vec

    def forward(self, interaction):
        geek_vec = self._single_bpj_layer(interaction, 'geek')
        job_vec = self._single_bpj_layer(interaction, 'job')

        x = torch.cat([job_vec, geek_vec, job_vec - geek_vec], dim=1)
        x = self.mlp(x).squeeze(1)
        return x

    def forward_neg(self, interaction):
        geek_vec = self._single_bpj_layer(interaction, 'geek')
        job_vec = self._single_bpj_layer(interaction, 'job', pre='neg_')

        x = torch.cat([job_vec, geek_vec, job_vec - geek_vec], dim=1)
        x = self.mlp(x).squeeze(1)
        return x

    def calculate_loss(self, interaction):
        label = interaction['label']
        output_pos = self.forward(interaction)
        output_neg_1 = self.forward_neg(interaction)
        
        label_pos = interaction['label_pos'].to(self.config['device']).squeeze()
        label_neg = interaction['label_neg'].to(self.config['device']).squeeze()
        
        return self.loss(output_pos, label_pos) \
                + self.loss(output_neg_1, label_neg) \

    def predict(self, interaction):
        # import pdb
        # pdb.set_trace()
        score = self.forward(interaction)
        return self.sigmoid(score)
