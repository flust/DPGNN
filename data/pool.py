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


class PopPool(PJFPool):
    def __init__(self, config):
        super(PopPool, self).__init__(config)


class MFPool(PJFPool):
    def __init__(self, config):
        super(MFPool, self).__init__(config)


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
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in tqdm(file):
                    #print(line)
                    try:
                        token, sent = line.strip().split('\t')
                    except:
                        #print(line)
                        continue
                    idx = token2id[token]
                    if idx not in sents:
                        sents[idx] = torch.zeros(tensor_size).long()
                        sent_num[idx] = 0
                    if sent_num[idx] == tensor_size[0]: continue
                    sent = torch.LongTensor([self.wd2id[_] if _ in self.wd2id else 1 for _ in sent.split(' ')])
                    sents[idx][sent_num[idx]] = F.pad(sent, (0, tensor_size[1] - len(sent)))
                    sent_num[idx] += 1
            setattr(self, f'{target}_sents', sents)


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


class VPJFv8Pool(VPJFPool):
    def __init__(self, config):
        super(VPJFv8Pool, self).__init__(config)
