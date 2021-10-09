import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
import re
import jieba
import numpy as np
import gzip
from torch.autograd import Variable
import torch.nn.functional as f
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import os
import torch.nn.utils.rnn  as rnn
import torch.nn.utils.rnn as rnn_utils

from model.abstract import PJFModel
# from model.layer import MLPLayers
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# max_pos_num = 2
# max_neg_num = 3
max_his_num = 5
max_sen_len = 10
doc_max_len = 5
vocab_size = 91168
hidden_dim = 100
dropout_rate = 0.2
# batch_size = 2048
# step_per_epoch = 800
# valid_size = 4096
# dir = './'
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.config = config
        self.sen_maxl = max_sen_len
        self.his_maxl = doc_max_len
        self.total_word = vocab_size + 2
        self.embedding_dim = config['embedding_size']
        self.embedding = nn.Embedding(self.total_word, self.embedding_dim, padding_idx=0)

    def forward(self, ids):
        # if max(ids) > 90000:
            # pdb.set_trace()
        # pdb.set_trace()
        ids = Variable(torch.LongTensor(ids), requires_grad=False).to(self.config['device']).view(-1, self.sen_maxl)
        emb = self.embedding(ids)
        length = self.get_length(ids)
        return emb, length

    def get_length(self, ids):
        length = (1 + torch.abs(2 * torch.sum((ids != 0), 1).long() - 1)) / 2
        return length.cpu()


class Sen_Rnn(nn.Module):
    def __init__(self):
        super(Sen_Rnn, self).__init__()
        self.sen_maxl = max_sen_len
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, emb, length):
        length_order, index_order = torch.sort(length, descending=True)
        emb_order = emb[index_order]
        reverse_order = torch.sort(index_order)[1]
        pack_emb = rnn_utils.pack_padded_sequence(emb_order, length_order, batch_first=True)
        out, hidden = self.rnn(pack_emb)
        hidden = hidden.squeeze(0)
        return hidden[reverse_order]


class Doc_Rnn(nn.Module):
    def __init__(self):
        super(Doc_Rnn, self).__init__()
        self.doc_maxl = doc_max_len
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=50, bidirectional=True)

    def forward(self, sen_rep, length):
        '''

        :param sen_rep: -1 * doc_len * hidden
        :return:
        '''
        length = torch.LongTensor(length).view(-1)
        length_order, index_order = torch.sort(length, descending=True)
        emb_order = sen_rep[index_order]
        reverse_order = torch.sort(index_order)[1]
        pack_emb = rnn_utils.pack_padded_sequence(emb_order, length_order, batch_first=True)
        out, hidden = self.rnn(pack_emb)
        out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True, total_length=self.doc_maxl)
        return out[reverse_order]


class Reading_memory(nn.Module):
    def __init__(self, config):
        super(Reading_memory, self).__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.doc_maxl = doc_max_len
        self.eps = -1e20
        self.transform_mem = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
                                           nn.Tanh())
        self.transform_sub = nn.Sequential(nn.Linear(in_features=2 * self.hidden_dim, out_features=self.hidden_dim),
                                           nn.Tanh())
        self.update_gate = nn.Sequential(nn.Linear(in_features=3 * self.hidden_dim, out_features=self.hidden_dim),
                                         nn.Sigmoid())

    def forward(self, sub_emb, memory, sub_len, sub_emb_raw):
        mem_4att = self.transform_mem(memory)
        sub_4att = self.transform_sub(torch.cat((sub_emb, sub_emb_raw), 2))
        sub_mem_att = torch.bmm(sub_4att, mem_4att.permute(0, 2, 1))
        mem_inf_mask, mem_zero_mask = self.get_mask(sub_len)
        sub_mem_att = mem_zero_mask * f.softmax((mem_inf_mask + sub_mem_att), 2)
        sub_mem = torch.bmm(sub_mem_att, memory)
        update_gate = self.update_gate(torch.cat((sub_emb, sub_mem, sub_mem * sub_emb), 2))
        sub_emb = (1 - update_gate) * sub_emb + update_gate * sub_mem
        return sub_emb

    def get_mask(self, length):
        bsz = len(length)

        inf_mask = torch.zeros(bsz, self.doc_maxl, self.doc_maxl).to(self.config['device'])
        zero_mask = torch.ones(bsz, self.doc_maxl, self.doc_maxl).to(self.config['device'])
        for i in range(bsz):
            if length[i] != self.doc_maxl:
                inf_mask[i, length[i]:, :] = self.eps
                zero_mask[i, length[i]:, :] = 0.0

        return inf_mask, zero_mask


class Updating_memory(nn.Module):
    def __init__(self, config):
        super(Updating_memory, self).__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.doc_maxl = doc_max_len
        self.eps = -1e20
        self.transform_mem = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
                                           nn.Tanh())
        self.transform_his = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
                                           nn.Tanh())
        self.transform_sub = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
                                           nn.Tanh())
        self.update_gate = nn.Sequential(nn.Linear(in_features=3 * self.hidden_dim, out_features=self.hidden_dim),
                                         nn.Sigmoid())

    def forward(self, his_emb, sub_emb, memory, sub_len, his_len):
        his_4att = self.transform_his(his_emb)
        sub_4att = self.transform_sub(sub_emb)
        memory_4att = self.transform_mem(memory)
        mem_his_att = torch.bmm(memory_4att, his_4att.permute(0, 2, 1))
        mem_sub_att = torch.bmm(memory_4att, sub_4att.permute(0, 2, 1))
        his_inf_mask, his_zero_mask = self.get_mask(his_len)
        sub_inf_mask, sub_zero_mask = self.get_mask(sub_len)
        mem_his_att = his_zero_mask * f.softmax((mem_his_att + his_inf_mask), 2)
        mem_his = torch.bmm(mem_his_att, his_emb)
        mem_sub_att = sub_zero_mask * f.softmax((mem_sub_att + sub_inf_mask), 2)
        mem_sub = torch.bmm(mem_sub_att, sub_emb)
        mem_att = mem_his + mem_sub
        update_gate = self.update_gate(torch.cat((mem_att, memory, mem_att * memory), 2))
        memory = (1 - update_gate) * memory + update_gate * mem_att
        return memory

    def get_mask(self, length):
        bsz = len(length)
        inf_mask = torch.zeros(bsz, self.doc_maxl, self.doc_maxl).to(self.config['device'])
        zero_mask = torch.ones(bsz, self.doc_maxl, self.doc_maxl).to(self.config['device'])
        for i in range(bsz):
            if length[i] != self.doc_maxl:
                inf_mask[i, length[i]:, :] = self.eps
                zero_mask[i, length[i]:, :] = 0.0
        return inf_mask, zero_mask


class Classfier(nn.Module):
    def __init__(self):
        super(Classfier, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.Bil = nn.Bilinear(in1_features=self.hidden_dim, in2_features=self.hidden_dim,
                               out_features=self.hidden_dim)
        self.Dropout_bi = nn.Dropout(dropout_rate)
        self.MLP = nn.Sequential(nn.Tanh(),
                                 nn.Linear(in_features=hidden_dim, out_features=1),
                                 nn.Sigmoid())
        self.Dropout_mlp = nn.Dropout(dropout_rate)

    def forward(self, sub, obj):
        feature = self.Bil(sub, obj)
        # feature = self.Dropout_bi(feature)
        match_score = self.MLP(feature).squeeze(1)
        return match_score


class JRMPM(PJFModel):
    def __init__(self, config, pool):
        super(JRMPM, self).__init__(config, pool)
        self.config = config

        self.embedding_dim = config['embedding_size']
        self.doc_maxl = doc_max_len
        self.sen_maxl = max_sen_len
        # self.valid_size = valid_size
        self.hidden_dim = hidden_dim
        self.his_len = max_his_num
        self.eps = 1e-30

        # self.pos_set = dt['pos_set']
        # self.neg_set = dt['neg_set']
        self.resume_id = pool.resume_id
        self.job_id = pool.job_id
        self.geek_dt = pool.geek_dt
        self.job_dt = pool.job_dt
        self.word_emb_dt = pool.wordemb
        
        self.Embedding = Embedding(config).to(config['device'])
        self.Embedding.embedding.weight.data.copy_(self.word_emb_dt)
        self.Job_sen_rnn = Sen_Rnn().to(config['device'])
        self.Geek_sen_rnn = Sen_Rnn().to(config['device'])
        self.Job_doc_rnn = Doc_Rnn().to(config['device'])
        self.Geek_doc_rnn = Doc_Rnn().to(config['device'])
        self.Job_r_memory = Reading_memory(config).to(config['device'])
        self.Geek_r_memory = Reading_memory(config).to(config['device'])
        self.Job_u_memory = Updating_memory(config).to(config['device'])
        self.Geek_u_memory = Updating_memory(config).to(config['device'])
        self.Classfier = Classfier().to(config['device'])
        # self.pos_train, self.pos_valid = self.divide_dataset(self.pos_set)
        # self.neg_train, self.neg_valid = self.divide_dataset(self.neg_set)
        self.optimizer = torch.optim.Adam(
            list(self.Job_doc_rnn.parameters())
            + list(self.Job_sen_rnn.parameters())
            + list(self.Geek_doc_rnn.parameters())
            + list(self.Geek_sen_rnn.parameters())
            + list(self.Job_r_memory.parameters())
            + list(self.Job_u_memory.parameters())
            + list(self.Geek_r_memory.parameters())
            + list(self.Geek_u_memory.parameters())
            + list(self.Classfier.parameters())
            , lr=5e-4)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([config['pos_weight']]))

    def parse_pair(self, batch_pair_geek, batch_pair_job):
        geek_content_id = []
        geek_content_length = []
        geek_history_id = []
        geek_history_length = []
        geek_history_num = []
        job_content_id = []
        job_content_length = []
        job_history_id = []
        job_history_length = []
        job_history_num = []
        for geek_id, job_id in zip(batch_pair_geek, batch_pair_job):
            # [geek_id, job_id] = pair
            # [job_id, geek_id] = pair
            geek_id, job_id = geek_id.item(), job_id.item()
            geek_dt = self.geek_dt[geek_id]
            geek_content_itemid = self.resume_id[geek_id]['id']
            geek_content_itemlength = self.resume_id[geek_id]['length']
            geek_history = geek_dt['history']

            geek_history_numitem = geek_dt['his_number']
            geek_history_itemids = []
            geek_history_itemlengths = []
            for id in geek_history:
                geek_history_itemid = self.job_id[id]['id']
                geek_history_itemlength = self.job_id[id]['length']
                geek_history_itemids.append(geek_history_itemid)
                geek_history_itemlengths.append(geek_history_itemlength)
            geek_content_id.append(geek_content_itemid)
            geek_content_length.append(geek_content_itemlength)
            geek_history_id.append(geek_history_itemids)
            geek_history_length.append(geek_history_itemlengths)
            geek_history_num.append(geek_history_numitem)

            job_dt = self.job_dt[job_id]
            job_content_itemid = self.job_id[job_id]['id']
            job_content_itemlength = self.job_id[job_id]['length']
            job_history = job_dt['history']
            job_history_numitem = job_dt['his_number']
            job_history_itemids = []
            job_history_itemlengths = []
            job_history_num.append(job_history_numitem)
            for id in job_history:
                job_history_itemid = self.resume_id[id]['id']
                job_history_itemlength = self.resume_id[id]['length']
                job_history_itemids.append(job_history_itemid)
                job_history_itemlengths.append(job_history_itemlength)
            job_content_id.append(job_content_itemid)
            job_content_length.append(job_content_itemlength)
            job_history_id.append(job_history_itemids)
            job_history_length.append(job_history_itemlengths)
        return geek_content_id, geek_content_length, geek_history_id, geek_history_length, geek_history_num, job_content_id, job_content_length, job_history_id, job_history_length, job_history_num

    def representation(self, sub, sub_len, his, his_len, sub_his_num, interaction_len, geek=True):

        # sub_len = torch.LongTensor(sub_len)
        # print(his_len)
        his_len = torch.LongTensor(his_len)

        # pdb.set_trace()

        sub_wordemb, sub_wordemb_len = self.Embedding(sub)  # batchsize * doclen, senlen, embedding
        his_wordemb, his_wordemb_len = self.Embedding(his)  # batchsize * hisnum * doclen * senlen * embedding

        sub_wordemb_3d = sub_wordemb.view(-1, self.sen_maxl, self.embedding_dim)
        his_wordemb_3d = his_wordemb.view(-1, self.sen_maxl, self.embedding_dim)
        # print(sub_wordemb_3d.size(), his_wordemb_3d.size())

        if geek:
            # pdb.set_trace()
            sub_senemb_raw = self.Geek_sen_rnn(sub_wordemb_3d, sub_wordemb_len).view(-1, self.doc_maxl, self.hidden_dim)
            # his_senemb = self.Job_sen_rnn(his_wordemb_3d, his_wordemb_len).view(-1,  self.doc_maxl, self.hidden_dim)
            his_senemb = self.Job_sen_rnn(his_wordemb_3d, his_wordemb_len).view(-1, self.his_len, self.doc_maxl,
                                                                                self.hidden_dim)
            sub_senemb = self.Geek_doc_rnn(sub_senemb_raw, sub_len)
            # his_senemb = self.Job_doc_rnn(his_senemb,his_len).view(-1, self.his_len,self.doc_maxl, self.hidden_dim)
            representation = []
            memory = sub_senemb
            # print(sub_senemb_raw.size(), sub_senemb.size())
            for i in range(self.his_len):
                his_item = his_senemb[:, i, :, :]
                his_item_len = his_len[:, i]
                # print(his_item.size(), his_item_len.size(), sub_senemb.size())
                memory = self.Geek_u_memory(his_item, sub_senemb, memory, sub_len, his_item_len)
                sub_senemb = self.Geek_r_memory(sub_senemb, memory, sub_len, sub_senemb_raw)
                representation.append(sub_senemb)
            representation = torch.stack(representation, 1)  ## batch * his_num * doc_len * hidden_dim
            representation_3d = representation.view(-1, self.doc_maxl, self.hidden_dim)
            representation_useful_index = []

            for i in range(interaction_len):
                index = i * self.his_len + (sub_his_num[i] - 1)
                representation_useful_index.append(index)
            # pdb.set_trace()
            representation_useful_index = torch.LongTensor(representation_useful_index).to(self.config['device'])
            representation_useful = torch.index_select(representation_3d, 0,
                                                       representation_useful_index)  ####  batch * doc * hidden_dim
            # pdb.set_trace()
            representation_useful = torch.max(representation_useful, 1)[0]  ##batch * hidden_dim
            return representation_useful

        if not geek:
            sub_senemb_raw = self.Job_sen_rnn(sub_wordemb_3d, sub_wordemb_len).view(-1, self.doc_maxl, self.hidden_dim)
            # his_senemb = self.Geek_sen_rnn(his_wordemb_3d, his_wordemb_len).view(-1, self.doc_maxl, self.hidden_dim)
            his_senemb = self.Geek_sen_rnn(his_wordemb_3d, his_wordemb_len).view(-1, self.his_len, self.doc_maxl,
                                                                                 self.hidden_dim)
            sub_senemb = self.Job_doc_rnn(sub_senemb_raw, sub_len)
            # his_senemb = self.Geek_doc_rnn(his_senemb, his_len).view(-1, self.his_len, self.doc_maxl, self.hidden_dim)
            representation = []
            memory = sub_senemb
            for i in range(self.his_len):
                his_item = his_senemb[:, i, :, :]
                his_item_len = his_len[:, i]
                memory = self.Job_u_memory(his_item, sub_senemb, memory, sub_len, his_item_len)
                sub_senemb = self.Job_r_memory(sub_senemb, memory, sub_len, sub_senemb_raw)
                representation.append(sub_senemb)
            representation = torch.stack(representation, 1)  ## batch * his_num * doc_len * hidden_dim
            representation_3d = representation.view(-1, self.doc_maxl, self.hidden_dim)
            representation_useful_index = []

            for i in range(interaction_len):
                index = i * self.his_len + (sub_his_num[i] - 1)
                representation_useful_index.append(index)
            representation_useful_index = torch.LongTensor(representation_useful_index).to(self.config['device'])
            # print("max():", max(representation_useful_index))
            # print("min():", min(representation_useful_index))
            # print("3d shape:", representation_3d.shape)
            representation_useful = torch.index_select(representation_3d, 0,
                                                       representation_useful_index)  ####  batch * doc * hidden_dim
            representation_useful = torch.max(representation_useful, 1)[0]
            return representation_useful

    def model(self, batch_pair_geek, batch_pair_job):
        interaction_len = batch_pair_geek.shape[0]

        geek_content_id, geek_content_length, geek_history_id, geek_history_length, geek_history_num, job_content_id, job_content_length, job_history_id, job_history_length, job_history_num = self.parse_pair(
            batch_pair_geek, batch_pair_job)

        geek_rep = self.representation(geek_content_id, geek_content_length, geek_history_id, geek_history_length,
                                       geek_history_num, interaction_len)
        
        job_rep = self.representation(job_content_id, job_content_length, job_history_id, job_history_length,
                                      job_history_num, interaction_len, geek=False)
        score = self.Classfier(geek_rep, job_rep)
        return score

    def forward(self, interaction):
        user = interaction['geek_id']
        item = interaction['job_id']
        # pdb.set_trace()
        score = self.model(user, item)
        return score

    def calculate_loss(self, interaction):
        label = interaction['label']
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
