import os
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import dynamic_load


def create_datasets(config, pool):
    return [
        dynamic_load(config, 'data.dataset', 'Dataset')(config, pool, phase)
        for phase in ['train', 'valid', 'test']
    ]


class PJFDataset(Dataset):
    def __init__(self, config, pool, phase):
        assert phase in ['train', 'test', 'valid']
        super(PJFDataset, self).__init__()
        self.config = config
        self.phase = phase
        self.logger = getLogger()

        self._init_attributes(pool)
        self._load_inters()

    def _init_attributes(self, pool):
        self.geek_num = pool.geek_num
        self.job_num = pool.job_num
        self.geek_token2id = pool.geek_token2id
        self.job_token2id = pool.job_token2id

    def _load_inters(self):
        filepath = os.path.join(self.config['dataset_path'], f'data.{self.phase}')
        self.logger.info(f'Loading from {filepath}')

        self.geek_ids, self.job_ids, self.labels = [], [], []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                geek_token, job_token, ts, label = line.strip().split('\t')
                geek_id = self.geek_token2id[geek_token]
                self.geek_ids.append(geek_id)
                job_id = self.job_token2id[job_token]
                self.job_ids.append(job_id)
                self.labels.append(int(label))
        self.geek_ids = torch.LongTensor(self.geek_ids)
        self.job_ids = torch.LongTensor(self.job_ids)
        self.labels = torch.FloatTensor(self.labels)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return {
            'geek_id': self.geek_ids[index],
            'job_id': self.job_ids[index],
            'label': self.labels[index]
        }

    def __str__(self):
        return '\n\t'.join([f'{self.phase} Dataset:'] + [
            f'{self.labels.shape[0]} interactions'
        ])

    def __repr__(self):
        return self.__str__()


class PopDataset(PJFDataset):
    def __init__(self, config, pool, phase):
        super(PopDataset, self).__init__(config, pool, phase)


class MFDataset(PJFDataset):
    def __init__(self, config, pool, phase):
        super(MFDataset, self).__init__(config, pool, phase)


class BPJFNNDataset(PJFDataset):
    def __init__(self, config, pool, phase):
        super(BPJFNNDataset, self).__init__(config, pool, phase)

    def _init_attributes(self, pool):
        super(BPJFNNDataset, self)._init_attributes(pool)
        self.geek_id2longsent = pool.geek_id2longsent
        self.geek_id2longsent_len = pool.geek_id2longsent_len
        self.job_id2longsent = pool.job_id2longsent
        self.job_id2longsent_len = pool.job_id2longsent_len

    def __getitem__(self, index):
        geek_id = self.geek_ids[index]
        job_id = self.job_ids[index]
        return {
            'geek_id': geek_id,
            'geek_longsent': self.geek_id2longsent[geek_id],
            'geek_longsent_len': self.geek_id2longsent_len[geek_id],
            'job_longsent': self.job_id2longsent[job_id],
            'job_longsent_len': self.job_id2longsent_len[job_id],
            'label': self.labels[index]
        }

class PJFNNDataset(PJFDataset):
    def __init__(self, config, pool, phase):
        super().__init__(config, pool, phase)

    def _init_attributes(self, pool):
        super()._init_attributes(pool)
        self.geek_sents = pool.geek_sents
        self.job_sents = pool.job_sents

    def __getitem__(self, index):
        geek_id = self.geek_ids[index]
        geek_id_item = geek_id.item()
        job_id = self.job_ids[index].item()
        return {
            'geek_id': geek_id,
            'geek_sents': self.geek_sents[geek_id_item],
            'job_sents': self.job_sents[job_id],
            'label': self.labels[index]
        }


class BERTDataset(PJFDataset):
    def __init__(self, config, pool, phase):
        super(BERTDataset, self).__init__(config, pool, phase)

    def _load_inters(self):
        super(BERTDataset, self)._load_inters()
        bert_filepath = os.path.join(self.config['dataset_path'], f'data.{self.phase}.bert.npy')
        self.logger.info(f'Loading from {bert_filepath}')
        self.bert_vec = torch.FloatTensor(np.load(bert_filepath).astype(np.float32))
        assert self.labels.shape[0] == self.bert_vec.shape[0]

    def __getitem__(self, index):
        return {
            'geek_id': self.geek_ids[index],
            'bert_vec': self.bert_vec[index],
            'label': self.labels[index]
        }

    def __str__(self):
        return '\n\t'.join([
            super(BERTDataset, self).__str__(),
            f'bert_vec: {self.bert_vec.shape}'
        ])


class RawVPJFDataset(PJFDataset):
    def __init__(self, config, pool, phase):
        super(RawVPJFDataset, self).__init__(config, pool, phase)
        np.save(os.path.join(self.config['dataset_path'], f'data.{self.phase}.job_his'), self.job_hiss.numpy())
        np.save(os.path.join(self.config['dataset_path'], f'data.{self.phase}.qwd_his'), self.qwd_hiss.numpy())
        np.save(os.path.join(self.config['dataset_path'], f'data.{self.phase}.qlen_his'), self.qlen_hiss.numpy())
        np.save(os.path.join(self.config['dataset_path'], f'data.{self.phase}.his_len'), self.qhis_len.numpy())

    def _init_attributes(self, pool):
        super(RawVPJFDataset, self)._init_attributes(pool)
        self.job_id2longsent = pool.job_id2longsent
        self.job_id2longsent_len = pool.job_id2longsent_len
        self.wd2id = pool.wd2id

    def _load_inters(self):
        query_his_filepath = os.path.join(self.config['dataset_path'], f'data.search.{self.phase}')
        self.logger.info(f'Loading from {query_his_filepath}')
        self.geek_ids, self.job_ids, self.labels = [], [], []
        self.job_hiss, self.qwd_hiss, self.qlen_hiss, self.qhis_len = [], [], [], []
        query_his_len = self.config['query_his_len']
        query_wd_len = self.config['query_wd_len']
        with open(query_his_filepath, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                geek_token, job_token, label, job_his, qwd_his, qlen_his = line.strip().split('\t')

                geek_id = self.geek_token2id[geek_token]
                self.geek_ids.append(geek_id)

                job_id = self.job_token2id[job_token]
                self.job_ids.append(job_id)

                self.labels.append(int(label))

                job_his = torch.LongTensor([self.job_token2id[_] for _ in job_his.split(' ')])
                self.job_hiss.append(F.pad(job_his, (0, query_his_len - job_his.shape[0])))

                qwd_his = qwd_his.split(' ')
                qwd_his_list = []
                for single_qwd in qwd_his:
                    single_qwd = torch.LongTensor([self.wd2id[_] if _ in self.wd2id else 1 for _ in single_qwd.split('|')])
                    qwd_his_list.append(F.pad(single_qwd, (0, query_wd_len - single_qwd.shape[0])))
                qwd_his = torch.stack(qwd_his_list)
                qwd_his = F.pad(qwd_his, (0, 0, 0, query_his_len - qwd_his.shape[0]))
                self.qwd_hiss.append(qwd_his)

                qlen_his = torch.FloatTensor(list(map(float, qlen_his.split(' '))))
                self.qlen_hiss.append(F.pad(qlen_his, (0, query_his_len - qlen_his.shape[0]), value=1))

                self.qhis_len.append(min(query_his_len, job_his.shape[0]))
        self.geek_ids = torch.LongTensor(self.geek_ids)
        self.job_ids = torch.LongTensor(self.job_ids)
        self.labels = torch.FloatTensor(self.labels)
        self.job_hiss = torch.stack(self.job_hiss)
        self.qwd_hiss = torch.stack(self.qwd_hiss)
        self.qlen_hiss = torch.stack(self.qlen_hiss)
        self.qhis_len = torch.FloatTensor(self.qhis_len)

        bert_filepath = os.path.join(self.config['dataset_path'], f'data.{self.phase}.bert.npy')
        self.logger.info(f'Loading from {bert_filepath}')
        self.bert_vec = torch.FloatTensor(np.load(bert_filepath).astype(np.float32))
        assert self.labels.shape[0] == self.bert_vec.shape[0]

    def __getitem__(self, index):
        items = super(RawVPJFDataset, self).__getitem__(index)
        items.update({
            'bert_vec': self.bert_vec[index],
            'job_his': self.job_hiss[index],
            'qwd_his': self.qwd_hiss[index],
            'qlen_his': self.qlen_hiss[index],
            'his_len': self.qhis_len[index]
        })
        return items


class VPJFDataset(BERTDataset):
    def __init__(self, config, pool, phase):
        super(VPJFDataset, self).__init__(config, pool, phase)

    def _init_attributes(self, pool):
        super(VPJFDataset, self)._init_attributes(pool)
        self.job_id2longsent = pool.job_id2longsent
        self.job_id2longsent_len = pool.job_id2longsent_len
        self.wd2id = pool.wd2id

    def _load_inters(self):
        super(VPJFDataset, self)._load_inters()
        attrs = [
            ('job_his', torch.LongTensor),
            ('qwd_his', torch.LongTensor),
            ('qlen_his', torch.FloatTensor),
            ('his_len', torch.FloatTensor)
        ]
        for attr, meth in attrs:
            his_filepath = os.path.join(self.config['dataset_path'], f'data.{self.phase}.{attr}.npy')
            self.logger.info(f'Loading from {his_filepath}')
            setattr(self, attr, meth(np.load(his_filepath)))
        query_his_len = self.config['query_his_len']
        query_wd_len = self.config['query_wd_len']
        assert query_his_len == self.job_his.shape[1]
        assert query_his_len == self.qwd_his.shape[1]
        assert query_his_len == self.qlen_his.shape[1]
        assert query_wd_len == self.qwd_his.shape[2]

    def __getitem__(self, index):
        items = super(VPJFDataset, self).__getitem__(index)
        job_id = self.job_ids[index]
        items.update({
            'job_id': job_id,
            'job_longsent': self.job_id2longsent[job_id],
            'job_longsent_len': self.job_id2longsent_len[job_id],
            'job_his': self.job_his[index],
            'qwd_his': self.qwd_his[index],
            'qlen_his': self.qlen_his[index],
            'his_len': self.his_len[index]
        })
        return items


class VPJFv9Dataset(VPJFDataset):
    def __init__(self, config, pool, phase):
        super(VPJFv9Dataset, self).__init__(config, pool, phase)
