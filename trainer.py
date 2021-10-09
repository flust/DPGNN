import os
from logging import getLogger
from time import time

import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
import wandb

from utils import ensure_dir, get_local_time, dict2device
from evaluator import Evaluator
import pdb


class Trainer(object):
    """The Trainer for training and evaluation strategies.

    Initializing the Trainer needs two parameters: `config` and `model`.
    - `config` records the parameters information for controlling training and evaluation,
    such as `learning_rate`, `epochs`, `eval_step` and so on.
    - `model` is the instantiated object of a Model Class.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

        self.logger = getLogger()
        self.learner = config['learner'].lower()
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -1
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.evaluator = Evaluator(config)

    def _build_optimizer(self):
        """Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        opt2method = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'adagrad': optim.Adagrad,
            'rmsprop': optim.RMSprop,
            'sparse_adam': optim.SparseAdam
        }

        if self.learner in opt2method:
            optimizer = opt2method[self.learner](filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx):
        """Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        total_loss = None
        iter_data = (
            tqdm(
                enumerate(train_data),
                total=len(train_data),
                desc=f"Train {epoch_idx:>5}",
            )
        )
        for batch_idx, interaction in iter_data:
            interaction = dict2device(interaction, self.device)
            self.optimizer.zero_grad()
            losses = self.model.calculate_loss(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            # loss.backward(retain_graph=True)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
        return total_loss

    def _valid_epoch(self, valid_data):
        """Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result, valid_result_str = self.evaluate(valid_data, load_best_model=False)
        wandb.log(valid_result)
        valid_score = valid_result[self.valid_metric]
        return valid_score, valid_result, valid_result_str

    def _save_checkpoint(self, epoch):
        """Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_model_file)

    def resume_checkpoint(self, resume_file):
        """Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file
        """
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            self.logger.warning('Architecture configuration given in config file is different from that of checkpoint. '
                                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config['loss_decimal_place'] or 4
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = 'train_loss%d: %.' + str(des) + 'f'
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += 'train loss:' + des % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, verbose=True, saved=True):
        """Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = 'Saving current: %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()

                valid_score, valid_result, valid_result_str = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = self._early_stopping(
                    valid_score, self.best_valid_score, self.cur_step, max_step=self.stopping_step)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result:' + valid_result_str
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = 'Saving current best: %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result

    def _early_stopping(self, value, best, cur_step, max_step):
        """validation-based early stopping

        Args:
            value (float): current result
            best (float): best result
            cur_step (int): the number of consecutive steps that did not exceed the best result
            max_step (int): threshold steps for stopping

        Returns:
            tuple:
            - float,
            best result after this step
            - int,
            the number of consecutive steps that did not exceed the best result after this step
            - bool,
            whether to stop
            - bool,
            whether to update
        """
        stop_flag = False
        update_flag = False
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
        return best, cur_step, stop_flag, update_flag

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None,
                 save_score=False, group='all', reverse=False):
        """Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            save_score (bool): Save .score file to running dir if ``True``. Defaults to ``False``.
            group (str): Which group to evaluate, can be ``all``, ``weak``, ``skilled``.

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        if not eval_data:
            return

        score_file = None
        if save_score:
            model_name = self.config['model']
            tag = 'job' if reverse else 'user'
            score_file = open(f'{model_name}.score.{tag}', 'w', encoding='utf-8')

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        batch_matrix_list = []
        iter_data = (
            tqdm(
                enumerate(eval_data),
                total=len(eval_data),
                desc=f"Evaluate   ",
            )
        )

        for batch_idx, batched_data in iter_data:
            interaction = batched_data
            scores = self.model.predict(dict2device(interaction, self.device))
            # if save_score:
            #     for gid, jid, label, score in zip(interaction['geek_id'], 
            #                                     interaction['job_id'],
            #                                     interaction['label'],
            #                                     scores):
            #         line = '\t'.join([str(gid.item()), 
            #                             str(jid.item()), 
            #                             str(label.item()), 
            #                             str(score.item())]) + '\n'
            #         score_file.write(line)

            batch_matrix = self.evaluator.collect(interaction, scores, reverse)
            batch_matrix_list.append(batch_matrix)
        result, result_str = self.evaluator.evaluate(batch_matrix_list, group)

        if save_score:
            score_file.close()

        return result, result_str
