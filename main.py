import argparse

from config import Config
from utils import init_seed

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, help='Model to test')

    args = parser.parse_args()
    return args

def main_process(model, config_dict=None):
    config = Config(model, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

if __name__ == "__main__":
    args = get_arguments()
    main_process(model=args.model)
