import os
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import dynamic_load
import random

def create_datasets(config, pool):
    data_list = []
    if config['pattern'] == 'geek':
        # train on user data 
        data_list.extend(['train_g', 'valid_g', 'test_g'])
    elif config['pattern'] == 'job':
        # train on job data 
        data_list.extend(['train_j', 'valid_j', 'test_j'])
    else:  
        # others: train on full data
        # data_list.extend(['train_all', 'valid_g', 'valid_j'])
        data_list.extend(['train_all_add', 'valid_g', 'valid_j'])

        # test set for geek & test set for job
        data_list.extend(['test_g', 'test_j'])

    return [
        dynamic_load(config, 'data.dataset', 'Dataset')(config, pool, phase)
        for phase in data_list
    ]


class PJFDataset(Dataset):
    def __init__(self, config, pool, phase):
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
                # geek_token, job_token, ts, label = line.strip().split('\t')
                geek_token, job_token, label = line.strip().split('\t')[:3]
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
        self.pool = pool
    
    def _load_inters(self):
        filepath = os.path.join(self.config['dataset_path'], f'data.{self.phase}')
        self.logger.info(f'Loading from {filepath}')

        self.geek_ids, self.job_ids, self.labels = [], [], []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                geek_token, job_token, label = line.strip().split('\t')[:3]
                if self.phase[0:5] == 'train' and label[0] == '0':  # 训练过程只保留正例
                    continue
                geek_id = self.geek_token2id[geek_token]
                self.geek_ids.append(geek_id)
                job_id = self.job_token2id[job_token]
                self.job_ids.append(job_id)
                self.labels.append(int(label))
        self.geek_ids = torch.LongTensor(self.geek_ids)
        self.job_ids = torch.LongTensor(self.job_ids)
        self.labels = torch.FloatTensor(self.labels)

    def __getitem__(self, index):
        geek_id = self.geek_ids[index]
        job_id = self.job_ids[index]
        neg_geek = random.randint(1, self.geek_num - 1)
        neg_job = random.randint(1, self.job_num - 1)

        while neg_job in self.pool.geek2jobs[geek_id]:
            neg_job = random.randint(1, self.job_num - 1)
        while neg_geek in self.pool.job2geeks[job_id]:
            neg_geek = random.randint(1, self.geek_num - 1)

        return {
            'geek_id': self.geek_ids[index],
            'job_id': self.job_ids[index],
            'neg_geek': neg_geek,
            'neg_job': neg_job,
            'label_pos': torch.Tensor([1]),
            'label_neg': torch.Tensor([0]),
            'label': self.labels[index]
        }

class NCFDataset(MFDataset):
    def __init__(self, config, pool, phase):
        super(NCFDataset, self).__init__(config, pool, phase)


class LightGCNDataset(MFDataset): 
    def __init__(self, config, pool, phase):
        super(LightGCNDataset, self).__init__(config, pool, phase)


class LightGCNaDataset(MFDataset): 
    def __init__(self, config, pool, phase):
        super(LightGCNaDataset, self).__init__(config, pool, phase)


class LightGCNaBERTDataset(MFDataset): 
    def __init__(self, config, pool, phase):
        super(LightGCNaBERTDataset, self).__init__(config, pool, phase)


class LightGCNbDataset(MFDataset): 
    def __init__(self, config, pool, phase):
        super(LightGCNbDataset, self).__init__(config, pool, phase)


class MultiGCNDataset(MFDataset):
    def __init__(self, config, pool, phase):
        super(MultiGCNDataset, self).__init__(config, pool, phase)


class MultiGCNsDataset(MFDataset):
    def __init__(self, config, pool, phase):
        super(MultiGCNsDataset, self).__init__(config, pool, phase)


class LightGCNalDataset(MFDataset): 
    def __init__(self, config, pool, phase):
        super(LightGCNalDataset, self).__init__(config, pool, phase)


class MultiGCNslDataset(MFDataset):
    def __init__(self, config, pool, phase):
        super(MultiGCNslDataset, self).__init__(config, pool, phase)


class MultiGCNsl1Dataset(MFDataset):
    def __init__(self, config, pool, phase):
        super(MultiGCNsl1Dataset, self).__init__(config, pool, phase)


class MultiGCNsl1l2Dataset(MFDataset):
    def __init__(self, config, pool, phase):
        super(MultiGCNsl1l2Dataset, self).__init__(config, pool, phase)


class BGPJFDataset(MFDataset):
    def __init__(self, config, pool, phase):
        super(BGPJFDataset, self).__init__(config, pool, phase)


class MultiGNNDataset(MFDataset):
    def __init__(self, config, pool, phase):
        super(MultiGNNDataset, self).__init__(config, pool, phase)


class BPJFNNDataset(PJFDataset):
    def __init__(self, config, pool, phase):
        super(BPJFNNDataset, self).__init__(config, pool, phase)
        self.pool = pool

    def _init_attributes(self, pool):
        super(BPJFNNDataset, self)._init_attributes(pool)
        self.geek_id2longsent = pool.geek_id2longsent
        self.geek_id2longsent_len = pool.geek_id2longsent_len
        self.job_id2longsent = pool.job_id2longsent
        self.job_id2longsent_len = pool.job_id2longsent_len

    def __getitem__(self, index):
        geek_id = self.geek_ids[index]
        job_id = self.job_ids[index]

        neg_geek = random.randint(1, self.geek_num - 1)
        neg_job = random.randint(1, self.job_num - 1)
        while neg_job in self.pool.geek2jobs[geek_id]:
            neg_job = random.randint(1, self.job_num - 1)
        while neg_geek in self.pool.job2geeks[job_id]:
            neg_geek = random.randint(1, self.geek_num - 1)

        return {
            'geek_id': geek_id,
            'geek_longsent': self.geek_id2longsent[geek_id],
            'geek_longsent_len': self.geek_id2longsent_len[geek_id],
            'job_id': job_id,
            'job_longsent': self.job_id2longsent[job_id],
            'job_longsent_len': self.job_id2longsent_len[job_id],
            'label': self.labels[index],
            'label_pos': torch.Tensor([1]),
            'label_neg': torch.Tensor([0]),
            'neg_geek': neg_geek,
            'neg_geek_longsent': self.geek_id2longsent[neg_geek],
            'neg_geek_longsent_len': self.geek_id2longsent_len[neg_geek],
            'neg_job': neg_job,
            'neg_job_longsent': self.job_id2longsent[neg_job],
            'neg_job_longsent_len': self.job_id2longsent_len[neg_job],
        }

class PJFNNDataset(PJFDataset):
    def __init__(self, config, pool, phase):
        super().__init__(config, pool, phase)
        self.pool = pool

    def _init_attributes(self, pool):
        super()._init_attributes(pool)
        self.geek_sents = pool.geek_sents
        self.job_sents = pool.job_sents

    def __getitem__(self, index):
        geek_id = self.geek_ids[index]
        geek_id_item = geek_id.item()
        job_id = self.job_ids[index].item()

        neg_geek = random.randint(1, self.geek_num - 1)
        neg_job = random.randint(1, self.job_num - 1)
        while neg_job in self.pool.geek2jobs[geek_id]:
            neg_job = random.randint(1, self.job_num - 1)
        while neg_geek in self.pool.job2geeks[job_id]:
            neg_geek = random.randint(1, self.geek_num - 1)

        return {
            'geek_id': geek_id,
            'job_id': job_id,
            'neg_geek': neg_geek,
            'neg_job': neg_job,
            'geek_sents': self.geek_sents[geek_id_item],
            'job_sents': self.job_sents[job_id],
            'neg_geek_sents': self.geek_sents[neg_geek],
            'neg_job_sents': self.job_sents[neg_job],
            'label_pos': torch.Tensor([1]),
            'label_neg': torch.Tensor([0]),
            'label': self.labels[index]
        }


class IPJFDataset(PJFNNDataset):
    def __init__(self, config, pool, phase):
        super(IPJFDataset, self).__init__(config, pool, phase)


class APJFNNDataset(PJFNNDataset):
    def __init__(self, config, pool, phase):
        super(APJFNNDataset, self).__init__(config, pool, phase)


class BERTDataset(MFDataset):
    def __init__(self, config, pool, phase):
        super(BERTDataset, self).__init__(config, pool, phase)

