import os
from logging import getLogger
import importlib

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PJFDataset(Dataset):
    def __init__(self, config, pool, phase):
        assert phase in ['train', 'test', 'valid']
        super(PJFDataset, self).__init__()
        self.config = config
        self.phase = phase
        self.logger = getLogger()

        self._init_attributes(pool)
        self._load_inters()
        self._reformat()

    def _init_attributes(self, pool):
        self.geek_num = pool['geek_num']
        self.job_num = pool['job_num']
        self.geek_token2id = pool['geek_token2id']
        self.job_token2id = pool['job_token2id']

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
        self.geek_ids = np.array(self.geek_ids)
        self.job_ids = np.array(self.job_ids)
        self.labels = np.array(self.labels)

    def _reformat(self):
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


class MFDataset(PJFDataset):
    def __init__(self, config, pool, phase):
        super().__init__(config, pool, phase)


def create_datasets(config, pool):
    module = importlib.import_module('data.dataset')
    return [
        getattr(module, config['model'] + 'Dataset')(config, pool, phase)
        for phase in ['train', 'valid', 'test']
    ]
