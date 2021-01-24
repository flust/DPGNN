import torch
import torch.nn as nn

from model.abstract import PJFModel
from model.layer import MLPLayers


class VPJFv3(PJFModel):
    def __init__(self, config, pool):

        super(VPJFv3, self).__init__(config, pool)

        self.wd_embedding_size = config['wd_embedding_size']
        self.user_embedding_size = config['user_embedding_size']
        self.bert_embedding_size = config['bert_embedding_size']
        self.hd_size = config['hidden_size']
        self.dropout = config['dropout']
        self.num_heads = config['num_heads']
        self.query_his_len = config['query_his_len']
        self.max_job_longsent_len = config['job_longsent_len']

        self.emb = nn.Embedding(pool.wd_num, self.wd_embedding_size, padding_idx=0)
        # self.geek_emb = nn.Embedding(self.geek_num, self.user_embedding_size)
        self.job_emb = nn.Embedding(self.job_num, self.user_embedding_size, padding_idx=0)

        self.text_matching_fc = nn.Linear(self.bert_embedding_size, self.hd_size)

        self.job_desc_attn_layer = nn.Linear(self.wd_embedding_size, 1)

        self.wq = nn.Linear(self.wd_embedding_size, self.hd_size, bias=False)
        self.wv = nn.Linear(self.user_embedding_size, self.hd_size, bias=False)

        self.attn_layer = nn.MultiheadAttention(
            embed_dim=self.hd_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            bias=False
        )

        self.intent_mlp = MLPLayers(
            layers=[self.user_embedding_size * 4, self.hd_size, self.hd_size],
            dropout=self.dropout,
            activation='tanh'
        )

        self.pre_mlp = MLPLayers(
            layers=[self.hd_size, self.hd_size, 1],
            dropout=self.dropout,
            activation='tanh'
        )

        self.self_attn_layer = nn.Linear(self.hd_size, 1)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([config['pos_weight']]))

    def _text_matching_layer(self, interaction):
        bert_vec = interaction['bert_vec']                      # (B, bertD)
        text_matching_vec = self.text_matching_fc(bert_vec)     # (B, H)
        return text_matching_vec

    def _intent_modeling_layer(self, interaction):
        job_longsent = interaction['job_longsent']
        job_longsent_len = interaction['job_longsent_len']
        job_desc_vec = self.emb(job_longsent)                   # (B, L, wordD)
        job_desc_mask = torch.arange(self.max_job_longsent_len, device=job_desc_vec.device) \
                           .expand(len(job_longsent_len), self.max_job_longsent_len) \
                           >= job_longsent_len.unsqueeze(1)
        job_desc_attn_weight = self.job_desc_attn_layer(job_desc_vec)
        job_desc_attn_weight = torch.masked_fill(job_desc_attn_weight, job_desc_mask.unsqueeze(-1), -10000)
        job_desc_attn_weight = torch.softmax(job_desc_attn_weight, dim=1)
        job_desc_vec = torch.sum(job_desc_attn_weight * job_desc_vec, dim=1)
        job_desc_vec = self.wq(job_desc_vec)                    # (B, H)

        job_id = interaction['job_id']                          # (B)
        job_id_vec = self.job_emb(job_id)                       # (B, idD)

        job_his = interaction['job_his']                        # (B, Q)
        job_his_vec = self.job_emb(job_his)                     # (B, Q, idD)
        job_his_vec = self.wv(job_his_vec)                      # (B, Q, H)

        qwd_his = interaction['qwd_his']                        # (B, Q, W)
        qlen_his = interaction['qlen_his']                      # (B, Q)
        qwd_his_vec = self.emb(qwd_his)                         # (B, Q, W, wordD)
        qwd_his_vec = torch.sum(qwd_his_vec, dim=2) / \
                      qlen_his.unsqueeze(-1)                    # (B, Q, wordD)
        qwd_his_vec = self.wq(qwd_his_vec)                      # (B, Q, H)

        his_len = interaction['his_len']                        # (B)
        key_padding_mask = torch.arange(self.query_his_len, device=his_len.device) \
                           .expand(len(his_len), self.query_his_len) \
                           >= his_len.unsqueeze(1)

        intent_vec, _ = self.attn_layer(
            query=job_desc_vec.unsqueeze(0),
            key=qwd_his_vec.transpose(1, 0),
            value=job_his_vec.transpose(1, 0),
            key_padding_mask=key_padding_mask,
            attn_mask=key_padding_mask.unsqueeze(1)
        )
        intent_vec = intent_vec.squeeze(0)                      # (B, idD)

        intent_modeling_vec = self.intent_mlp(torch.cat([
            job_id_vec, intent_vec, job_id_vec - intent_vec, job_id_vec * intent_vec
        ], dim=1))                                              # (B, H)

        search_his_mask = torch.sum(job_his, dim=1) == 0

        return intent_modeling_vec, search_his_mask

    def predict_layer(self, vec1, vec2, mask2):
        a1 = self.self_attn_layer(vec1)                         # (B, 1)
        a2 = self.self_attn_layer(vec2)                         # (B, 1)
        a2 = torch.masked_fill(a2, mask2.unsqueeze(-1), -10000)
        weight = torch.softmax(torch.cat([a1, a2], dim=1), dim=1).unsqueeze(-1)
        vecs = torch.stack([vec1, vec2]).transpose(1, 0)
        x = torch.sum(weight * vecs, dim=1)
        score = self.pre_mlp(x).squeeze(-1)
        return score

    def forward(self, interaction):
        text_matching_vec = self._text_matching_layer(interaction)
        intent_modeling_vec, search_his_mask = self._intent_modeling_layer(interaction)
        score = self.predict_layer(text_matching_vec, intent_modeling_vec, search_his_mask)
        return score

    def calculate_loss(self, interaction):
        label = interaction['label']
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
