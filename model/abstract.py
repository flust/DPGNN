from logging import getLogger

import numpy as np
import torch.nn as nn


class PJFModel(nn.Module):
    r"""Base class for all Person-Job Fit models
    """

    def __init__(self, config, pool):
        super(PJFModel, self).__init__()

        self.logger = getLogger()
        self.device = config['device']

        self.geek_num = pool.geek_num
        self.job_num = pool.job_num

    def calculate_loss(self, interaction):
        """Calculate the training loss for a batch data.

        Args:
            interaction (dict): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        """Predict the scores between users and items.

        Args:
            interaction (dict): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def __str__(self):
        """Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(PJFModel, self).__str__() + '\n\tTrainable parameters: {}'.format(params)
