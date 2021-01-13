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


class MFPool(PJFPool):
    def __init__(self, config):
        super().__init__(config)


class MFwBERTPool(PJFPool):
    def __init__(self, config):
        super().__init__(config)
        self._load_bert_vec()

    def _load_bert_vec(self):
        for target in ['geek', 'job']:
            filepath = os.path.join(self.config['dataset_path'], f'{target}.bert.npy')
            self.logger.info(f'Loading {filepath}')
            bert_vec = np.load(filepath).astype(np.float32)
            setattr(self, f'{target}_bert_vec', bert_vec)

    def __str__(self):
        return '\n\t'.join([
            super().__str__(),
            f'geek_bert_vec: {self.geek_bert_vec.shape}',
            f'job_bert_vec: {self.job_bert_vec.shape}'
        ])
