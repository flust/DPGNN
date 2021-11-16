import os
from logging import getLogger
from typing import DefaultDict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import random
import pdb
from scipy.sparse import coo_matrix


class PJFPool(object):
    def __init__(self, config):
        self.logger = getLogger()
        self.config = config

        self._load_ids()

    def _load_ids(self):
        for target in ['geek', 'job']:
            token2id = {}
            id2token = []
            filepath = os.path.join(self.config['dataset_path'], f'{target}.token')
            self.logger.info(f'Loading {filepath}')
            with open(filepath, 'r') as file:
                for i, line in enumerate(file):
                    token = line.strip()
                    token2id[token] = i
                    id2token.append(token)
            setattr(self, f'{target}_token2id', token2id)
            setattr(self, f'{target}_id2token', id2token)
            setattr(self, f'{target}_num', len(id2token))

    def __str__(self):
        return '\n\t'.join(['Pool:'] + [
            f'{self.geek_num} geeks',
            f'{self.job_num} jobs'
        ])

    def __repr__(self):
        return self.__str__()


class PopPool(PJFPool):
    def __init__(self, config):
        super(PopPool, self).__init__(config)

# class MultiGCNPool(PJFPool):
#     def __init__(self, config):
#         super(MultiGCNPool, self).__init__(config)
#         self.success_edge = self._get_edge(os.path.join(self.config['dataset_path'], f'data.train_all'))
#         self.user_add_edge = self._get_edge(os.path.join(self.config['dataset_path'], f'data.user_add'))
#         self.job_add_edge = self._get_edge(os.path.join(self.config['dataset_path'], f'data.job_add'))

#     def _get_edge(self, filepath):
#         self.geek_ids, self.job_ids, self.labels = [], [], []

#         with open(filepath, 'r', encoding='utf-8') as file:
#             for line in tqdm(file):
#                 geek_token, job_token, label = line.strip().split('\t')[:3]
#                 if(int(label) == 0): continue
#                 geek_id = self.geek_token2id[geek_token]
#                 self.geek_ids.append(geek_id)
#                 job_id = self.job_token2id[job_token]
#                 self.job_ids.append(job_id)
#                 self.labels.append(int(label))

#         self.geek_ids = torch.LongTensor(self.geek_ids)
#         self.job_ids = torch.LongTensor(self.job_ids)
#         self.labels = torch.FloatTensor(self.labels)

#         return [self.geek_ids, self.job_ids]

# class BGPJFPool(MultiGCNPool):
#     def __init__(self, config):
#         super(BGPJFPool, self).__init__(config)
#         self.sample_n = config['sample_n']
#         self._load_neg()
#         if(config['ADD_BERT']):
#             self._load_bert()

#     def _load_neg(self):
#         # save neg sample
#         self.geek2jobs_neg = torch.zeros(self.geek_num, 1000)
#         self.geek2jobs_neg_num = DefaultDict(int)
#         self.job2geeks_neg = torch.zeros(self.job_num, 1000)
#         self.job2geeks_neg_num = DefaultDict(int)

#         data_all = open(os.path.join(self.config['dataset_path'], f'data.train_all'))
#         for l in tqdm(data_all):
#             gid, jid, label = l.split('\t')
#             if label == '1\n':
#                 continue
#             gid = self.geek_token2id[gid]
#             jid = self.job_token2id[jid]

#             self.geek2jobs_neg[gid][self.geek2jobs_neg_num[gid]] = jid
#             self.geek2jobs_neg_num[gid] += 1
#             self.job2geeks_neg[jid][self.job2geeks_neg_num[gid]] = gid
#             self.job2geeks_neg_num[jid] += 1

#     def _load_bert(self):
#         u_filepath = os.path.join(self.config['dataset_path'], 'geek.bert.npy')
#         self.logger.info(f'Loading from {u_filepath}')
#         j_filepath = os.path.join(self.config['dataset_path'], 'job.bert.npy')
#         # bert_filepath = os.path.join(self.config['dataset_path'], f'data.{self.phase}.bert.npy')
#         self.logger.info(f'Loading from {j_filepath}')

#         u_array = np.load(u_filepath).astype(np.float64)
#         # add padding 
#         u_array = np.vstack([u_array, np.zeros((1, u_array.shape[1]))])

#         j_array = np.load(j_filepath).astype(np.float64)
#         # add padding
#         j_array = np.vstack([j_array, np.zeros((1, j_array.shape[1]))])

#         self.geek_token2bertid = {}
#         self.job_token2bertid = {}
#         for i in range(u_array.shape[0]):
#             self.geek_token2bertid[str(u_array[i, 0].astype(int))] = i
#         for i in range(j_array.shape[0]):
#             self.job_token2bertid[str(j_array[i, 0].astype(int))] = i

#         self.u_bert_vec = torch.FloatTensor(u_array[:, 1:])
#         self.j_bert_vec = torch.FloatTensor(j_array[:, 1:])


class MFPool(PJFPool):
    def __init__(self, config):
        super(MFPool, self).__init__(config)
        self._load_inter()

    def _load_inter(self):
        self.geek2jobs = DefaultDict(list)
        self.job2geeks = DefaultDict(list)
        self.geek2neg = DefaultDict(list)
        self.job2neg = DefaultDict(list)

        data_all = open(os.path.join(self.config['dataset_path'], f'data.train_all'))
        for l in tqdm(data_all):
            gid, jid, label = l.split('\t')
            gid = self.geek_token2id[gid]
            jid = self.job_token2id[jid]
            if label[0] == '1':
                self.geek2jobs[gid].append(jid)
                self.job2geeks[jid].append(gid)
            else:
                self.geek2neg[gid].append(jid)
                self.job2neg[jid].append(gid)
            

class NCFPool(MFPool):
    def __init__(self, config):
        super(NCFPool, self).__init__(config)


class LightGCNPool(MFPool):
    def __init__(self, config):
        super(LightGCNPool, self).__init__(config)
        self.addfriend_in_graph = self.config['addfriend_in_graph']
        self._load_edge()
    
    def _load_edge(self):
        if(self.addfriend_in_graph):
            filepath = os.path.join(self.config['dataset_path'], f'data.train_all_add')
        else:
            filepath = os.path.join(self.config['dataset_path'], f'data.train_all')

        self.logger.info(f'Loading from {filepath}')

        self.geek_ids, self.job_ids, self.labels = [], [], []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                geek_token, job_token, label = line.strip().split('\t')[:3]

                geek_id = self.geek_token2id[geek_token]
                job_id = self.job_token2id[job_token]

                self.geek_ids.append(geek_id)
                self.job_ids.append(job_id)
                self.labels.append(int(label))
                # self.labels.append(1)

        self.geek_ids = torch.LongTensor(self.geek_ids)
        self.job_ids = torch.LongTensor(self.job_ids)
        self.labels = torch.FloatTensor(self.labels)
        
        src = self.geek_ids[self.labels == 1]
        tgt = self.job_ids[self.labels == 1]
        data = self.labels[self.labels == 1]
        self.interaction_matrix = coo_matrix((data, (src, tgt)), shape=(self.geek_num, self.job_num))


class MultiGCNPool(MFPool):
    def __init__(self, config):
        super(MultiGCNPool, self).__init__(config)
        success_file = os.path.join(self.config['dataset_path'], f'data.train_all')
        user_add_file = os.path.join(self.config['dataset_path'], f'data.user_add')
        job_add_file = os.path.join(self.config['dataset_path'], f'data.job_add')

        add_sample_rate = config['add_sample_rate']
        self.interaction_matrix = self._load_edge(success_file)
        self.user_add_matrix = self._load_edge(user_add_file, sample_rate=add_sample_rate)
        self.job_add_matrix = self._load_edge(job_add_file, sample_rate=add_sample_rate)

    def _load_edge(self, filepath, sample_rate=1):
        self.logger.info(f'Loading from {filepath}')
        self.geek_ids, self.job_ids, self.labels = [], [], []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                geek_token, job_token, label = line.strip().split('\t')[:3]

                geek_id = self.geek_token2id[geek_token]
                job_id = self.job_token2id[job_token]
                if random.random() <= sample_rate:
                    self.geek_ids.append(geek_id)
                    self.job_ids.append(job_id)
                    self.labels.append(int(label))
                # self.labels.append(1)

        self.geek_ids = torch.LongTensor(self.geek_ids)
        self.job_ids = torch.LongTensor(self.job_ids)
        self.labels = torch.FloatTensor(self.labels)
        
        src = self.geek_ids
        tgt = self.job_ids
        data = self.labels
        interaction_matrix = coo_matrix((data, (src, tgt)), shape=(self.geek_num, self.job_num))
        return interaction_matrix


class SingleBERTPool(PJFPool):
    def __init__(self, config):
        super(SingleBERTPool, self).__init__(config)
        self._load_bert_vec()

    def _load_bert_vec(self):
        for target in ['geek', 'job']:
            filepath = os.path.join(self.config['dataset_path'], f'{target}.bert.npy')
            self.logger.info(f'Loading {filepath}')
            bert_vec = np.load(filepath).astype(np.float32)
            setattr(self, f'{target}_bert_vec', bert_vec)

    def __str__(self):
        return '\n\t'.join([
            super(SingleBERTPool, self).__str__(),
            f'geek_bert_vec: {self.geek_bert_vec.shape}',
            f'job_bert_vec: {self.job_bert_vec.shape}'
        ])


class BERTPool(PJFPool):
    def __init__(self, config):
        super(BERTPool, self).__init__(config)


class MultiPJFPool(MultiGCNPool):
    def __init__(self, config):
        super(MultiPJFPool, self).__init__(config)
        self.sample_n = config['sample_n']
        self._load_group()
        self._load_bert()

    def _load_group(self):
        self.geek2jobs = {}
        self.job2geeks = {}
        data_all = open(os.path.join(self.config['dataset_path'], f'data.train_all'))
        for l in data_all:
            gid, jid, label = l.split('\t')
            if label == '0\n':
                continue
            gid = self.geek_token2id[gid]
            jid = self.job_token2id[jid]
            if gid not in self.geek2jobs.keys():
                self.geek2jobs[gid] = [jid]
            else:
                self.geek2jobs[gid].append(jid)
            if jid not in self.job2geeks.keys():
                self.job2geeks[jid] = [gid]
            else:
                self.job2geeks[jid].append(gid)

    def _load_ids(self):
        for target in ['geek', 'job']:
            token2id = {}
            id2token = []
            filepath = os.path.join(self.config['dataset_path'], f'{target}.token')
            self.logger.info(f'Loading {filepath}')
            with open(filepath, 'r') as file:
                for i, line in enumerate(file):
                    token = line.strip()
                    token2id[token] = i
                    id2token.append(token)
            # add padding
            token2id['0'] = len(id2token)
            id2token.append('0')

            setattr(self, f'{target}_token2id', token2id)
            setattr(self, f'{target}_id2token', id2token)
            setattr(self, f'{target}_num', len(id2token))

    def _load_bert(self):
        u_filepath = os.path.join(self.config['dataset_path'], 'geek.bert.npy')
        self.logger.info(f'Loading from {u_filepath}')
        j_filepath = os.path.join(self.config['dataset_path'], 'job.bert.npy')
        # bert_filepath = os.path.join(self.config['dataset_path'], f'data.{self.phase}.bert.npy')
        self.logger.info(f'Loading from {j_filepath}')

        u_array = np.load(u_filepath).astype(np.float64)
        # add padding 
        u_array = np.vstack([u_array, np.zeros((1, u_array.shape[1]))])

        j_array = np.load(j_filepath).astype(np.float64)
        # add padding
        j_array = np.vstack([j_array, np.zeros((1, j_array.shape[1]))])

        self.geek_token2bertid = {}
        self.job_token2bertid = {}
        for i in range(u_array.shape[0]):
            self.geek_token2bertid[str(u_array[i, 0].astype(int))] = i
        for i in range(j_array.shape[0]):
            self.job_token2bertid[str(j_array[i, 0].astype(int))] = i
        # import pdb
        # pdb.set_trace()
        self.u_bert_vec = torch.FloatTensor(u_array[:, 1:])
        self.j_bert_vec = torch.FloatTensor(j_array[:, 1:])


class BPJFNNPool(PJFPool):
    def __init__(self, config):
        super(BPJFNNPool, self).__init__(config)
        self._load_word_cnt()
        self._load_longsent()

    def _load_word_cnt(self):
        min_word_cnt = self.config['min_word_cnt']
        self.wd2id = {
            '[WD_PAD]': 0,
            '[WD_MISS]': 1
        }
        self.id2wd = ['[WD_PAD]', '[WD_MISS]']
        filepath = os.path.join(self.config['dataset_path'], 'word.cnt')
        self.logger.info(f'Loading {filepath}')
        with open(filepath, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                wd, cnt = line.strip().split('\t')
                if int(cnt) < min_word_cnt:
                    break
                self.wd2id[wd] = i + 2
                self.id2wd.append(wd)
        self.wd_num = len(self.id2wd)

    def _load_longsent(self):
        for target in ['geek', 'job']:
            max_sent_len = self.config[f'{target}_longsent_len']
            id2longsent = torch.zeros([getattr(self, f'{target}_num'), max_sent_len], dtype=torch.int64)
            id2longsent_len = torch.zeros(getattr(self, f'{target}_num'))
            filepath = os.path.join(self.config['dataset_path'], f'{target}.longsent')
            token2id = getattr(self, f'{target}_token2id')
            self.logger.info(f'Loading {filepath}')
            err = 0
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in tqdm(file):
                    # import pdb
                    # pdb.set_trace()
                    try:
                        token, longsent = line.strip().split('\t')
                    except:
                        token = line.strip()
                        longsent = '[WD_PAD]'
                    if token not in token2id:
                        err += 1
                        continue
                    idx = token2id[token]
                    longsent = torch.LongTensor([self.wd2id[_] if _ in self.wd2id else 1 for _ in longsent.split(' ')])
                    id2longsent[idx] = F.pad(longsent, (0, max_sent_len - longsent.shape[0]))
                    id2longsent_len[idx] = min(max_sent_len, longsent.shape[0])
                    
            setattr(self, f'{target}_id2longsent', id2longsent)
            setattr(self, f'{target}_id2longsent_len', id2longsent_len)
            

    def __str__(self):
        return '\n\t'.join([
            super(BPJFNNPool, self).__str__(),
            f'{self.wd_num} words',
            f'geek_id2longsent: {self.geek_id2longsent.shape}',
            f'job_id2longsent: {self.job_id2longsent.shape}'
        ])

class PJFNNPool(PJFPool):
    def __init__(self, config):
        super().__init__(config)
        # pdb.set_trace()
        self._load_word_cnt()
        self._load_sents()

    def _load_word_cnt(self):
        min_word_cnt = self.config['min_word_cnt']
        self.wd2id = {
            '[WD_PAD]': 0,
            '[WD_MISS]': 1
        }
        self.id2wd = ['[WD_PAD]', '[WD_MISS]']
        filepath = os.path.join(self.config['dataset_path'], 'word.cnt')
        self.logger.info(f'Loading {filepath}')
        with open(filepath, 'r', encoding='utf-8') as file:
            for i, line in tqdm(enumerate(file)):
                wd, cnt = line.strip().split('\t')
                if int(cnt) < min_word_cnt:
                    break
                self.wd2id[wd] = i + 2
                self.id2wd.append(wd)
        self.wd_num = len(self.id2wd)

    def _load_sents(self):
        for target in ['geek', 'job']:
            filepath = os.path.join(self.config['dataset_path'], f'{target}.sent')
            max_sent_num = self.config[f'{target}_max_sent_num']
            max_sent_len = self.config[f'{target}_max_sent_len']

            sents = {}
            sent_num = {}
            tensor_size = [max_sent_num, max_sent_len]
            token2id = getattr(self, f'{target}_token2id')
            self.logger.info(f'Loading {filepath}')

            # 看一下平均句子数量和平均句子长度
            sent_num_sum = 0  # 句子数量  
            sent_len_sum = 0  # 句子长度
            user_num_count = 0  # 
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in tqdm(file):
                    try:
                        token, sent = line.strip().split('\t')
                    except:
                        continue
                    if token not in token2id:
                        continue
                    idx = token2id[token]
                    if idx not in sents:
                        sents[idx] = torch.zeros(tensor_size).long()
                        sent_num[idx] = 0
                        user_num_count += 1
                    if sent_num[idx] == tensor_size[0]: continue
                    sent_len_sum += len(sent.split(' '))
                    sent = torch.LongTensor([self.wd2id[_] if _ in self.wd2id else 1 for _ in sent.split(' ')])
                    sents[idx][sent_num[idx]] = F.pad(sent, (0, tensor_size[1] - len(sent)))   # sents[idx] 第idx个用户的多个句子组成的 tensor 矩阵
                    sent_num[idx] += 1    # sent_num[idx] 第idx个用户的句子个数
                    sent_num_sum += 1

            # pdb.set_trace()
            avg_sent_num = sent_num_sum / user_num_count   # 2.606
            avg_sent_len = sent_len_sum / sent_num_sum   # 13.939
            
            setattr(self, f'{target}_sents', sents)
            setattr(self, f'{target}_sent_num', sent_num)
            
class IPJFPool(PJFNNPool):
    def __init__(self, config):
        super(IPJFPool, self).__init__(config)

class APJFNNPool(PJFNNPool):
    def __init__(self, config):
        super(APJFNNPool, self).__init__(config)

class JRMPMPool(PJFNNPool):
    def __init__(self, config):
        super(JRMPMPool, self).__init__(config)
        self._load_word_emb()
        self._load_jrmpm_ids()
        
    def _load_word_emb(self):
        self.wordemb = np.load(os.path.join(self.config['dataset_path'], 'word.emb.npy'))
        z = np.zeros((2, self.wordemb.shape[1]))
        self.wordemb = np.vstack((z, self.wordemb))
        self.wordemb = torch.FloatTensor(self.wordemb)

    def _load_sents(self):
        for target in ['geek', 'job']:
            filepath = os.path.join(self.config['dataset_path'], f'{target}.sent')
            max_sent_num = self.config[f'{target}_max_sent_num']
            max_sent_len = self.config[f'{target}_max_sent_len']

            sents = {}
            sent_num = {}
            tensor_size = [max_sent_num, max_sent_len]
            token2id = getattr(self, f'{target}_token2id')
            self.logger.info(f'Loading {filepath}')
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in tqdm(file):
                    try:
                        token, sent = line.strip().split('\t')
                    except:
                        continue
                    if token not in token2id:
                        continue
                    idx = token2id[token]
                    if idx not in sents:
                        sents[idx] = torch.zeros(tensor_size).long()
                        sent_num[idx] = 0
                    if sent_num[idx] == tensor_size[0]: continue
                    sent = torch.LongTensor([self.wd2id[_] if _ in self.wd2id else 1 for _ in sent.split(' ')])
                    sents[idx][sent_num[idx]] = F.pad(sent, (0, tensor_size[1] - len(sent)))   # sents[idx] 第idx个用户的多个句子组成的 tensor 矩阵
                    sent_num[idx] += 1    # sent_num[idx] 第idx个用户的句子个数
            sents['0'] = torch.zeros(tensor_size).long()
            sent_num['0'] = 0
            setattr(self, f'{target}_sents', sents)
            setattr(self, f'{target}_sent_num', sent_num)

    def _load_jrmpm_ids(self):
        f = open(os.path.join(self.config['dataset_path'], 'data.train_all'))
        self.job_dt = {}
        self.geek_dt = {}
        for l in f:
            gtoken, jtoken, label = l.split('\t')
            if label == '0\n':
                continue
            gid = self.geek_token2id[gtoken]
            jid = self.job_token2id[jtoken]

            if jid not in self.job_dt.keys():
                self.job_dt[jid] = {
                    'history': [gid],
                    'his_number': 1
                }
            self.job_dt[jid]['history'].append(gid)
            self.job_dt[jid]['his_number'] += 1

            if gid not in self.geek_dt.keys():
                self.geek_dt[gid] = {
                    'history': [jid],
                    'his_number': 1
                }
            self.geek_dt[gid]['history'].append(jid)
            self.geek_dt[gid]['his_number'] += 1

        self.job_id = {}
        self.resume_id = {}
        for jid in self.job_sents.keys():
            self.job_id[jid] = {
                'id': self.job_sents[jid].tolist(),
                'length': self.job_sent_num[jid]
            }
            
            if jid not in self.job_dt.keys():
                self.job_dt[jid] = {
                    'history': ['0'] * 10,
                    'his_number': 1
                }
            elif len(self.job_dt[jid]['history']) < 10:
                self.job_dt[jid]['history'].extend(['0'] * (10 - len(self.job_dt[jid]['history'])))
            else:
                self.job_dt[jid]['history'] = self.job_dt[jid]['history'][:10]
                self.job_dt[jid]['his_number'] = 10

        for gid in self.geek_sents.keys():
            self.resume_id[gid] = {
                'id': self.geek_sents[gid].tolist(),
                'length': self.geek_sent_num[gid]
            }

            if gid not in self.geek_dt.keys():
                self.geek_dt[gid] = {
                    'history': ['0'] * 10,
                    'his_number': 1
                }
            elif len(self.geek_dt[gid]['history']) < 10:
                self.geek_dt[gid]['history'].extend(['0'] * (10 - len(self.geek_dt[gid]['history'])))
            else:
                self.geek_dt[gid]['history'] = self.geek_dt[gid]['history'][:10]
                self.geek_dt[gid]['his_number'] = 10


class VPJFPool(BPJFNNPool):
    def __init__(self, config):
        super(VPJFPool, self).__init__(config)

    def _load_ids(self):
        super(VPJFPool, self)._load_ids()
        filepath = os.path.join(self.config['dataset_path'], 'job.search.token')
        self.logger.info(f'Loading {filepath}')
        ori_job_num = self.job_num
        with open(filepath, 'r') as file:
            for i, line in enumerate(file):
                token = line.strip()
                assert token not in self.job_token2id
                self.job_token2id[token] = i + ori_job_num
                self.job_id2token.append(token)
        self.job_search_token_num = len(self.job_id2token) - ori_job_num
        self.job_num = len(self.job_id2token)

    def _load_word_cnt(self):
        super(VPJFPool, self)._load_word_cnt()
        ori_wd_num = len(self.id2wd)
        filepath = os.path.join(self.config['dataset_path'], 'word.search.id')
        self.logger.info(f'Loading {filepath}')
        with open(filepath, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                wd = line.strip()
                assert wd not in self.wd2id
                self.wd2id[wd] = i + ori_wd_num
                self.id2wd.append(wd)
        self.search_wd_num = len(self.id2wd) - ori_wd_num
        self.wd_num = len(self.id2wd)

    def __str__(self):
        return '\n\t'.join([
            super(VPJFPool, self).__str__(),
            f'{self.job_search_token_num} job tokens only exist in search log',
            f'{self.search_wd_num} words only exist in search log'
        ])


class VPJFv9Pool(VPJFPool):
    def __init__(self, config):
        super(VPJFv9Pool, self).__init__(config)


