import os
from logging import getLogger


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
        return self.pool[item] if item in self.pool else None

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
