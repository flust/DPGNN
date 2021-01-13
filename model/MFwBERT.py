import torch
import torch.nn as nn

from model.MF import MF


class MFwBERT(MF):
    def __init__(self, config, pool):

        super(MFwBERT, self).__init__(config, pool)

        # define layers and loss
        self.geek_emb = nn.Embedding.from_pretrained(
            torch.from_numpy(pool.geek_bert_vec),
            freeze=False,
            padding_idx=0
        )
        self.job_emb = nn.Embedding.from_pretrained(
            torch.from_numpy(pool.job_bert_vec),
            freeze=False,
            padding_idx=0
        )
