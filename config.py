import os
import re

import yaml
import torch

class Config(object):
    """ Configurator module that load the defined parameters.

    Configurator module will first load parameters from ``prop/overall.yaml`` and ``prop/[model].yaml``,
    then load parameters from ``config_dict``

    The priority order is as following:

    config dict > yaml config file
    """

    def __init__(self, model, config_dict=None):
        """
        Args:
            model (str): the model name.
            config_dict (dict): the external parameter dictionary, default is None.
        """
        self.parameters = dict()
        self.yaml_loader = self._build_yaml_loader()
        self.parameters.update(self._load_config_files(model))
        self.parameters.update(config_dict)
        self._init_device()

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        return loader

    def _load_config_files(self, model):
        file_config_dict = dict()
        file_list = [
            'prop/overall.yaml',
            f'prop/{model}.yaml'
        ]
        for file in file_list:
            with open(file, 'r', encoding='utf-8') as f:
                file_config_dict.update(yaml.load(f.read(), Loader=self.yaml_loader))
        return file_config_dict

    def _init_device(self):
        use_gpu = self.parameters['use_gpu']
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.parameters['gpu_id'])
        self.parameters['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def __getitem__(self, item):
        if item in self.parameters:
            return self.parameters[item]
        else:
            return None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.parameters

    def __str__(self):
        return '\n'.join([
            f'{arg}={value}'
            for arg, value in self.parameters.items()
        ]) + '\n\n'

    def __repr__(self):
        return self.__str__()

