import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn.init import normal_

from model.abstract import PJFModel
from model.layer import FusionLayer


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
        # if method is 'max':
        #     self.pool = nn.AdaptiveMaxPool2d((1, dim))
        # elif method is 'mean':
        #     self.pool = nn.AdaptiveAvgPool2d((1, dim))
        # else:
        #     raise ValueError('method {} not exist'.format(method))

    def forward(self, x):
        x = self.net1(x)  # [2048, 20, 13, 64]
        x = self.net2(x).squeeze(2) # [2048, 20, 64]
        # x = self.pool(x).squeeze(1) # [2048, 64]
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

        self.job_attn = nn.Linear(self.embedding_size, 1)
        self.geek_attn = nn.Linear(self.embedding_size, 1)

        self.geek_fusion_layer = FusionLayer(self.embedding_size)
        self.job_fusion_layer = FusionLayer(self.embedding_size)

        self.w_job = nn.Linear(2 * self.embedding_size, 1)
        self.w_geek = nn.Linear(2 * self.embedding_size, 1)

        # self.matching_mlp = nn.Linear(4 * self.embedding_size, 1)

        self.matching_mlp = nn.Sequential(
            nn.Linear(4 * self.embedding_size, 2 * self.embedding_size),
            nn.BatchNorm1d(num_features=2 * self.embedding_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2 * self.embedding_size, 1)
        )
        # =====

        self.pool = nn.AdaptiveMaxPool2d((1, self.embedding_size))
        self.mlp = nn.Linear(self.embedding_size, 1)
        # =====
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        self.apply(self._init_weights)

    def forward(self, interaction):
        geek_sents = interaction['geek_sents']
        job_sents = interaction['job_sents']
        geek_vec = self.emb(geek_sents)
        job_vec = self.emb(job_sents)
        
        geek_matrix = self.geek_layer(geek_vec) # [2048, 15, 64]
        job_matrix = self.job_layer(job_vec)  # [2048, 20, 64]
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
        geek_attn = self.geek_attn(geek_matrix)
        geek_attn = torch.softmax(geek_attn, dim=1)
        geek_vec = torch.sum(geek_matrix * geek_attn, dim=1)

        job_attn = self.geek_attn(job_matrix)
        job_attn = torch.softmax(job_attn, dim=1)
        job_vec = torch.sum(job_matrix * job_attn, dim=1)

        # import pdb
        # pdb.set_trace()
        # fusion layer
        geek_f = self.geek_fusion_layer(geek_vec, job_vec)
        job_f = self.job_fusion_layer(geek_vec, job_vec)

        geek_r = self.w_geek(geek_f)  # [2048, 1]
        job_r = self.w_job(job_f)  # [2048, 1]
        # =====
        # import pdb
        # pdb.set_trace()
        
        # x = geek_vec * job_vec

        # x = self.mlp(x).squeeze(1)
        # return x    
        # =====
        x = self.matching_mlp(torch.cat((geek_f, job_f), 1))
        return x.squeeze(1)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        if isinstance(module, nn.Parameter):
            normal_(module.weight.data, 0, 0.01)

    def calculate_loss(self, interaction):
        label = interaction['label']
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return torch.sigmoid(self.forward(interaction))