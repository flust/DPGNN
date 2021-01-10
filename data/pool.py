import os
from logging import getLogger

from tqdm import tqdm
import numpy as np


class PJFPool(object):
    def __init__(self, config):
        self.logger = getLogger()
        self.config = config

        self.pool = {}
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
            self.pool[f'{target}_token2id'] = token2id
            self.pool[f'{target}_id2token'] = id2token
            self.pool[f'{target}_num'] = len(id2token)

    def __getitem__(self, item):
        return self.pool[item]

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.pool

    def __str__(self):
        return '\n\t'.join(['Pool:'] + [
            f'{self.pool["geek_num"]} geeks',
            f'{self.pool["job_num"]} jobs'
        ])

    def __repr__(self):
        return self.__str__()


class MFPool(PJFPool):
    def __init__(self, config):
        super().__init__(config)


class MFwBERTPool(PJFPool):
    def __init__(self, config):
        super().__init__(config)
        self._load_bert_vec()

    def _load_bert_vec(self):
        for target in ['geek', 'job']:
            bert_vec = np.zeros([self.pool[f'{target}_num'], self.config['embedding_size']])
            token2id = self.pool[f'{target}_token2id']
            filepath = os.path.join(self.config['dataset_path'], f'{target}.bert')
            self.logger.info(f'Loading {filepath}')
            with open(filepath, 'r') as file:
                for line in tqdm(file):
                    token, vec = line.strip().split('\t')
                    idx = token2id[token]
                    vec = np.array(list(map(float, vec.split(' '))))
                    assert vec.shape[0] == self.config['embedding_size']
                    bert_vec[idx] = vec
            self.pool[f'{target}_bert_vec'] = bert_vec

    def __str__(self):
        return '\n\t'.join([
            super().__str__(),
            f'geek_bert_vec: {self.pool["geek_bert_vec"].shape}',
            f'job_bert_vec: {self.pool["job_bert_vec"].shape}'
        ])
