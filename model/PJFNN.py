import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from model.abstract import PJFModel


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
        if method == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, dim))
        elif method == 'mean':
            self.pool = nn.AdaptiveAvgPool2d((1, dim))
        else:
            raise ValueError('method {} not exist'.format(method))

    def forward(self, x):
        # import pdb
        # pdb.set_trace()   # [2048, 20, 30, 64]
        x = self.net1(x)  # [2048, 20, 13, 64]
        x = self.net2(x).squeeze(2) # [2048, 20, 64]
        x = self.pool(x).squeeze(1) # [2048, 64]
        return x

class PJFNN(PJFModel):
    def __init__(self, config, pool):
        super(PJFNN, self).__init__(config, pool)
        self.embedding_size = config['embedding_size']
        self.geek_channels = config['geek_max_sent_num'] #20
        self.job_channels = config['job_max_sent_num']   #20
        self.config = config

        # define layers and loss
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

        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, 1)
        )
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, geek_sents, job_sents):
        geek_vec = self.emb(geek_sents)
        job_vec = self.emb(job_sents)
        geek_vec = self.geek_layer(geek_vec)
        job_vec = self.job_layer(job_vec)
        x = geek_vec * job_vec
        x = self.mlp(x).squeeze(1)
        return x

    def calculate_loss(self, interaction):
        label = interaction['label']
        geek_sents = interaction['geek_sents']
        job_sents = interaction['job_sents']
        neg_job_sents = interaction['neg_job_sents']

        output_pos = self.forward(geek_sents, job_sents)
        output_neg_1 = self.forward(geek_sents, neg_job_sents)
        label_pos = interaction['label_pos'].to(self.config['device']).squeeze()
        label_neg = interaction['label_neg'].to(self.config['device']).squeeze()
        
        loss = self.loss(output_pos, label_pos)
        loss += self.loss(output_neg_1, label_neg)
        return loss

    def predict(self, interaction):
        geek_sents = interaction['geek_sents']
        job_sents = interaction['job_sents']
        return torch.sigmoid(self.forward(geek_sents, job_sents))