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
        self._load_inter()

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

    def _load_inter(self):
        self.geek2jobs = DefaultDict(list)
        self.job2geeks = DefaultDict(list)

        data_all = open(os.path.join(self.config['dataset_path'], f'data.train_all_add'))
        for l in tqdm(data_all):
            gid, jid, label = l.split('\t')
            gid = self.geek_token2id[gid]
            jid = self.job_token2id[jid]
            self.geek2jobs[gid].append(jid)
            self.job2geeks[jid].append(gid)

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


class MFPool(PJFPool):
    def __init__(self, config):
        super(MFPool, self).__init__(config)


class LFRRPool(MFPool):
    def __init__(self, config):
        super(LFRRPool, self).__init__(config)
            
       
class NCFPool(MFPool):
    def __init__(self, config):
        super(NCFPool, self).__init__(config)


class LightGCNPool(MFPool):
    def __init__(self, config):
        super(LightGCNPool, self).__init__(config)
        success_file = os.path.join(self.config['dataset_path'], f'data.train_all')
        self.interaction_matrix = self._load_edge(success_file)

    def _load_edge(self, filepath):
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

        self.geek_ids = torch.LongTensor(self.geek_ids)
        self.job_ids = torch.LongTensor(self.job_ids)
        self.labels = torch.FloatTensor(self.labels)
        
        src = self.geek_ids
        tgt = self.job_ids
        data = self.labels
        interaction_matrix = coo_matrix((data, (src, tgt)), shape=(self.geek_num, self.job_num))
        return interaction_matrix


class LightGCNaPool(LightGCNPool):
    def __init__(self, config):
        super(LightGCNaPool, self).__init__(config)
        user_add_file = os.path.join(self.config['dataset_path'], f'data.user_add')
        job_add_file = os.path.join(self.config['dataset_path'], f'data.job_add')

        # add_sample_rate = config['add_sample_rate']
        self.user_add_matrix = self._load_edge(user_add_file)
        self.job_add_matrix = self._load_edge(job_add_file)


class DPGNNPool(LightGCNaPool):
    def __init__(self, config):
        super(DPGNNPool, self).__init__(config)
        if(config['ADD_BERT']):
            self._load_bert()
    
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

        self.u_bert_vec = torch.FloatTensor(u_array[:, 1:])
        self.j_bert_vec = torch.FloatTensor(j_array[:, 1:])


class LightGCNaBERTPool(DPGNNPool):
    def __init__(self, config):
        super(LightGCNaBERTPool, self).__init__(config)


class BPJFNNPool(PJFPool):
    def __init__(self, config):
        super(BPJFNNPool, self).__init__(config)
        # import pdb
        # pdb.set_trace()
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
                    # import pdb
                    # pdb.set_trace()
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
                    # sent_num_sum += 1

            # avg_sent_num = sent_num_sum / user_num_count   # 1.8  /  11.9
            # avg_sent_len = sent_len_sum / sent_num_sum   # 13.1  /  12.4
            # sales user 1.8 * 13.1
            # sales job 11.9 * 12.4
            # tech user 2.7 * 14.1
            # tech job 10.5 * 11.8

            setattr(self, f'{target}_sents', sents)
            setattr(self, f'{target}_sent_num', sent_num)


class APJFNNPool(PJFNNPool):
    def __init__(self, config):
        super(APJFNNPool, self).__init__(config)


class BERTPool(PJFPool):
    def __init__(self, config):
        super(BERTPool, self).__init__(config)
        self._load_bert()
    
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

        self.u_bert_vec = torch.FloatTensor(u_array[:, 1:])
        self.j_bert_vec = torch.FloatTensor(j_array[:, 1:])


class IPJFPool(PJFNNPool):
    def __init__(self, config):
        super(IPJFPool, self).__init__(config)
        self._load_bert()

    def _load_inter(self):
        self.geek2jobs = DefaultDict(list)
        self.geek2jobs_addfriend = DefaultDict(list)
        
        self.job2geeks = DefaultDict(list)
        self.job2geeks_addfriend = DefaultDict(list)
        data_all = open(os.path.join(self.config['dataset_path'], f'data.train_all_add'))
        for l in tqdm(data_all):
            gid, jid, label = l.split('\t')
            gid = self.geek_token2id[gid]
            jid = self.job_token2id[jid]
            if label == '1\n':
                self.geek2jobs[gid].append(jid)
                self.job2geeks[jid].append(gid)
            elif label == '2\n':
                self.geek2jobs_addfriend[gid].append(jid)
            else:
                self.job2geeks_addfriend[jid].append(gid)

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

        self.u_bert_vec = torch.FloatTensor(u_array[:, 1:])
        self.j_bert_vec = torch.FloatTensor(j_array[:, 1:])


class PJFFFPool(BERTPool):
    def __init__(self, config):
        super(PJFFFPool, self).__init__(config)



class woBGPool(DPGNNPool):
    def __init__(self, config):
        super(woBGPool, self).__init__(config)

class woBLPool(DPGNNPool):
    def __init__(self, config):
        super(woBLPool, self).__init__(config)

class woMLPool(DPGNNPool):
    def __init__(self, config):
        super(woMLPool, self).__init__(config)

class woBERTPool(DPGNNPool):
    def __init__(self, config):
        super(woBERTPool, self).__init__(config)
