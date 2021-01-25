import argparse
from logging import getLogger

import torch
from torch.utils.data import DataLoader

from trainer import Trainer
from utils import init_seed, init_logger, dynamic_load


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, help='Model file to test.')
    parser.add_argument('--phase', '-p', default='test', help='Which phase to evaluate.')
    parser.add_argument('--save', '-s', action='store_true', help='Whether to save predict score.')

    args = parser.parse_args()
    return args


def eval_process(resume_file, phase='test', save=False):
    assert phase in ['train', 'test', 'valid']
    checkpoint = torch.load(resume_file)

    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # data preparation
    pool = dynamic_load(config, 'data.pool', 'Pool')(config)
    logger.info(pool)

    eval_dataset = dynamic_load(config, 'data.dataset', 'Dataset')(config, pool, phase)
    logger.info(eval_dataset)

    eval_data = DataLoader(
        dataset=eval_dataset,
        batch_size=config['eval_batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    # model loading and initialization
    model = dynamic_load(config, 'model')(config, pool).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model evaluation
    _, eval_result = trainer.evaluate(eval_data, load_best_model=False,
                                   show_progress=config['show_progress'], save_score=save)

    logger.info('result: {}'.format(eval_result))


if __name__ == "__main__":
    args = get_arguments()
    eval_process(resume_file=args.file, save=args.save)
