import argparse
from logging import getLogger
import wandb
import os

from config import Config
from data.dataset import create_datasets
from data.dataloader import construct_dataloader
from trainer import Trainer
from utils import init_seed, init_logger, dynamic_load

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, help='Model to test.', default='BGPJF')
    parser.add_argument('--name', '-n', type=str, help='Name of this run.')
    parser.add_argument('--direction', '-d', type=str, help='direction to evaluate', default='multi')
    parser.add_argument('--embedding_size', '-es', type=int, help='embedding size')
    parser.add_argument('--learning_rate', '-lr', type=float, help='learning rate')
    parser.add_argument('--dropout', '-do', type=float, help='dropout', default=0.2)
    parser.add_argument('--gpu_id', '-g', type=int, help='gpu_id', default=0)
    parser.add_argument('--n_layers', '-nl', type=int, help='n_layers', default=2)
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
    run = wandb.init(
        config=config.params,
        project='vpjf-0127',
        name=model if config['name'] is None else config['name'],
        reinit=True,
        mode='disabled'
    )

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

    # load dataset
    train_data, valid_data_g, valid_data_j, test_data_g, test_data_j = construct_dataloader(config, datasets)
        
    # model loading and initialization
    model = dynamic_load(config, 'model')(config, pool).to(config['device'])
    logger.info(model)
    wandb.watch(model, model.loss, log="all", log_freq=100)

    # trainer loading and initialization
    trainer = Trainer(config, model)
    
    # model training
    best_valid_score, best_valid_result_g, best_valid_result_j = trainer.fit(train_data, valid_data_g, valid_data_j, saved=saved)
    logger.info('best valid result for geek: {}'.format(best_valid_result_g))
    logger.info('best valid result for job: {}'.format(best_valid_result_j))

    # import pdb
    # pdb.set_trace()
    # model evaluation for user
    test_result, test_result_str = trainer.evaluate(test_data_g, load_best_model=True)
    wandb.log(test_result)
    logger.info('test for user result [all]: {}'.format(test_result_str))

    # model evaluation for job
    test_result, test_result_str = trainer.evaluate(test_data_j, load_best_model=True, reverse=True)
    wandb.log(test_result)
    logger.info('test for job result [all]: {}'.format(test_result_str))

    run.finish()

    return {
        'best_valid_score': best_valid_score,
        'best_valid_result_g': best_valid_result_g,
        'best_valid_result_j':best_valid_result_j,
        'test_result': test_result
    }


if __name__ == "__main__":
    args = get_arguments()    
    main_process(model=args.model, config_dict={
        'name': args.name,
        'direction': args.direction,
        'embedding_size': args.embedding_size,
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'gpu_id': args.gpu_id,
        'n_layers': args.n_layers
    })
