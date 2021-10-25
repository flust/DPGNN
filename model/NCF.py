import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn.init import normal_

from model.abstract import PJFModel
from model.layer import MLPLayers

class NCF(PJFModel):

    def __init__(self, config, pool):
        super(NCF, self).__init__(config, pool)
        self.config = config 
        self.pool = pool
        # load parameters info
        # self.mf_embedding_size = config['mf_embedding_size']
        # self.mlp_embedding_size = config['mlp_embedding_size']
        self.mf_embedding_size = config['embedding_size']
        self.mlp_embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout']
        self.mf_train = config['mf_train']
        self.mlp_train = config['mlp_train']

        # define layers and loss
        self.n_users = self.geek_num
        self.n_items = self.job_num
        self.user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.n_users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(self.n_items, self.mlp_embedding_size)
        
        # bert part
        self.ADD_BERT = config['ADD_BERT']
        self.BERT_e_size = 0
        if self.ADD_BERT:
            self.BERT_e_size = config['BERT_output_size']
            self.bert_lr = nn.Linear(config['BERT_embedding_size'],
                        self.BERT_e_size).to(self.config['device'])
            self._load_bert()
        self.mf_embedding_size += self.BERT_e_size
        self.mlp_embedding_size += self.BERT_e_size

        self.mlp_layers = MLPLayers([2 * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout_prob)
        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size + self.mlp_hidden_size[-1], 1)
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([config['pos_weight']]))

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)

        if self.ADD_BERT:
            self.bert_u = self.bert_lr(self.bert_user)
            self.bert_j = self.bert_lr(self.bert_job)
            # import pdb
            # pdb.set_trace()
            user_mf_e = torch.cat([user_mf_e, self.bert_u[user]], dim=1)
            item_mf_e = torch.cat([item_mf_e, self.bert_j[item]], dim=1)
            user_mlp_e = torch.cat([user_mlp_e, self.bert_u[user]], dim=1)
            item_mlp_e = torch.cat([item_mlp_e, self.bert_j[item]], dim=1)

        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
        if self.mlp_train:
            mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))  # [batch_size, layers[-1]]
        if self.mf_train and self.mlp_train:
            output = self.sigmoid(self.predict_layer(torch.cat((mf_output, mlp_output), -1)))
        elif self.mf_train:
            output = self.sigmoid(self.predict_layer(mf_output))
        elif self.mlp_train:
            output = self.sigmoid(self.predict_layer(mlp_output))
        else:
            raise RuntimeError('mf_train and mlp_train can not be False at the same time')

        return output.squeeze()

    def calculate_loss(self, interaction):
        user = interaction['geek_id']
        item = interaction['job_id']
        label = interaction['label']

        output = self.forward(user, item)
        return self.loss(output, label)

    def predict(self, interaction):
        user = interaction['geek_id']
        item = interaction['job_id']
        return self.forward(user, item)

    def _load_bert(self):
        self.bert_user = torch.FloatTensor([]).to(self.config['device'])
        for i in range(self.pool.geek_num):
            geek_token = self.pool.geek_id2token[i]
            bert_id =  self.pool.geek_token2bertid[geek_token]
            bert_u_vec = self.pool.u_bert_vec[bert_id, :].unsqueeze(0).to(self.config['device'])
            self.bert_user = torch.cat([self.bert_user, bert_u_vec], dim=0)

        self.bert_job = torch.FloatTensor([]).to(self.config['device'])
        for i in range(self.pool.job_num):
            job_token = self.pool.job_id2token[i]
            bert_id =  self.pool.job_token2bertid[job_token]
            bert_j_vec = self.pool.j_bert_vec[bert_id].unsqueeze(0).to(self.config['device'])
            self.bert_job = torch.cat([self.bert_job, bert_j_vec], dim=0)