import argparse
from logging import getLogger

from data.dataset import create_datasets
from config import Config
from utils import init_seed, init_logger, dynamic_load


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, help='Model to test')

    args = parser.parse_args()
    return args

def main_process(model, config_dict=None):
    """Main process API for experiments of VPJF

    Args:
        model (str): model name
        config_dict (dict): parameters dictionary used to modify experiment parameters
    """

    # configurations initialization
    config = Config(model, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # data preparation
    pool = dynamic_load(config, 'data.pool', 'Pool')
    logger.info(pool)
    datasets = create_datasets(config, pool)
    for ds in datasets:
        logger.info(ds)

if __name__ == "__main__":
    args = get_arguments()
    main_process(model=args.model)
