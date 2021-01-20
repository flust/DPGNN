import argparse
from logging import getLogger

import wandb

from config import Config
from data.dataset import create_datasets
from data.dataloader import construct_dataloader
from trainer import Trainer
from utils import init_seed, init_logger, dynamic_load


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, help='Model to test')
    parser.add_argument('--name', '-n', type=str, help='Name of exp')

    args = parser.parse_args()
    return args


def main_process(model, config_dict=None, saved=True):
    """Main process API for experiments of VPJF

    Args:
        model (str): Model name.
        config_dict (dict): Parameters dictionary used to modify experiment parameters.
            Defaults to ``None``.
        saved (bool): Whether to save the model parameters. Defaults to ``True``.
    """

    # configurations initialization
    config = Config(model, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # data preparation
    pool = dynamic_load(config, 'data.pool', 'Pool')(config)
    logger.info(pool)

    datasets = create_datasets(config, pool)
    for ds in datasets:
        logger.info(ds)

    train_data, valid_data, test_data = construct_dataloader(config, datasets)

    # model loading and initialization
    model = dynamic_load(config, 'model')(config, pool).to(config['device'])
    logger.info(model)
    wandb.watch(model, model.loss, log="all", log_freq=100)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=saved,
                                                      show_progress=config['show_progress'])

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved,
                                   show_progress=config['show_progress'])
    wandb.log(test_result)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))

    return {
        'best_valid_score': best_valid_score,
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == "__main__":
    args = get_arguments()
    main_process(model=args.model, config_dict={
        'name': args.name
    })
