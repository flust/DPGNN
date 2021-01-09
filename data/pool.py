class AbstractPool(object):
    def __init__(self, config):
        self.pool = {}

    def __getitem__(self, item):
        return self.pool[item] if item in self.pool else None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.pool

    def __str__(self):
        raise NotImplementedError()

    def __repr__(self):
        return self.__str__()

class MFPool(AbstractPool):
    def __init__(self, config):
        super().__init__(config)

    def __str__(self):
        return 'MF Pool contains nothing.'
