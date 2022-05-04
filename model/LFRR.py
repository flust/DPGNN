import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from model.abstract import PJFModel
from model.layer import BPRLoss, HingeLoss

class LFRR(PJFModel):
    def __init__(self, config, pool):

        super(LFRR, self).__init__(config, pool)
        self.config = config
        self.pool = pool
        self.embedding_size = config['embedding_size']
        self.n_users = self.pool.geek_num
        self.n_items = self.pool.job_num
        self.user_embedding_1 = nn.Embedding(self.n_users, self.embedding_size)
        self.user_embedding_2 = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding_1 = nn.Embedding(self.n_items, self.embedding_size)
        self.item_embedding_2 = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()
        self.sigmoid = nn.Sigmoid()
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward_ui(self, user, item):
        u_1 = self.user_embedding_1(user)
        i_1 = self.item_embedding_1(item)

        s_ui = torch.mul(u_1, i_1).sum(dim=1)
        return s_ui

    def forward_iu(self, user, item):
        u_2 = self.user_embedding_2(user)
        i_2 = self.item_embedding_2(item)

        s_iu = torch.mul(u_2, i_2).sum(dim=1)
        return s_iu

    def forward(self, user, item):
        s_ui = self.forward_ui(user, item)
        s_iu = self.forward_iu(user, item)
        score = s_ui + s_iu
        return score

    def calculate_loss(self, interaction):
        pos_user = interaction['geek_id']
        pos_item = interaction['job_id']
        neg_item = interaction['neg_job']
        neg_user = interaction['neg_geek']

        score_pos = self.forward(pos_user, pos_item)
        score_neg_1 = self.forward(pos_user, neg_item)
        score_neg_2 = self.forward(neg_user, pos_item)

        loss = self.loss(score_pos, score_neg_1) + self.loss(score_pos, score_neg_2)
        return loss

    def predict(self, interaction):
        user = interaction['geek_id']
        item = interaction['job_id']
        return self.forward(user, item)

