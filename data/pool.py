import os
from logging import getLogger

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F


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


class MFPool(PJFPool):
    def __init__(self, config):
        super(MFPool, self).__init__(config)


class MFwBERTPool(PJFPool):
    def __init__(self, config):
        super(MFwBERTPool, self).__init__(config)
        self._load_bert_vec()

    def _load_bert_vec(self):
        for target in ['geek', 'job']:
            filepath = os.path.join(self.config['dataset_path'], f'{target}.bert.npy')
            self.logger.info(f'Loading {filepath}')
            bert_vec = np.load(filepath).astype(np.float32)
            setattr(self, f'{target}_bert_vec', bert_vec)

    def __str__(self):
        return '\n\t'.join([
            super(MFwBERTPool, self).__str__(),
            f'geek_bert_vec: {self.geek_bert_vec.shape}',
            f'job_bert_vec: {self.job_bert_vec.shape}'
        ])


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
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in tqdm(file):
                    token, longsent = line.strip().split('\t')
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


class BERTPool(PJFPool):
    def __init__(self, config):
        super().__init__(config)
