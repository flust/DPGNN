import argparse
from logging import getLogger

import torch
from torch.utils.data import DataLoader

from trainer import Trainer
from utils import init_seed, init_logger, dynamic_load


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, help='Model file to test')

    args = parser.parse_args()
    return args


def test_process(resume_file):
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

    test_dataset = dynamic_load(config, 'data.dataset', 'Dataset')(config, pool, 'test')
    logger.info(test_dataset)

    test_data = DataLoader(
        dataset=test_dataset,
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
    test_result = trainer.evaluate(test_data, load_best_model=False,
                                   show_progress=config['show_progress'])

    logger.info('test result: {}'.format(test_result))


if __name__ == "__main__":
    args = get_arguments()
    test_process(resume_file=args.file)
