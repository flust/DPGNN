from copy import deepcopy
from logging import getLogger

# import wandb

from config import Config
from data.dataset import create_datasets
from data.dataloader import construct_dataloader
from trainer import Trainer
from utils import init_seed, init_logger, dynamic_load
import itertools
import wandb

MODEL = 'MF'
metrics = ['gauc', 'p@5', 'r@5', 'mrr']
all_params = {
    'embedding_size': [8, 16],
    # 'embedding_size': [128, 256],
    # 'n_layers': [2, 3],
    # 'sample_n': [3, 5, 8],
    # 'lambda_1': [0.01, 0.1, 0.5],
    # 'lambda_2': [0.01, 0.1, 0.5],
    # 'learning_rate': [0.01, 0.001, 0.0003],
}

def hyper_opt_preparation(model):
    # configurations initialization
    config = Config(model)
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

    train_data, valid_data_g, valid_data_j, test_data_g, test_data_j = construct_dataloader(config, datasets)
    
    return {
        'config': config,
        'pool': pool,
        'train_data': train_data,
        'valid_data_g': valid_data_g,
        'valid_data_j': valid_data_j,
        'test_data_g': test_data_g,
        'test_data_j': test_data_j,
    }


def single_run(config, pool, train_data, valid_data_g, valid_data_j, test_data_g, test_data_j):
    logger = getLogger()

    # model loading and initialization
    model = dynamic_load(config, 'model')(config, pool).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result_g, best_valid_result_j = trainer.fit(train_data, valid_data_g, valid_data_j, saved=True)
    logger.info('best valid result for geek: {}'.format(best_valid_result_g))
    logger.info('best valid result for job: {}'.format(best_valid_result_j))

    # model evaluation for user
    test_result_g, test_result_str_g = trainer.evaluate(test_data_g, load_best_model=True)
    # wandb.log(test_result_g)
    logger.info('test for user result [all]: {}'.format(test_result_str_g))

    # model evaluation for job
    test_result_j, test_result_str_j = trainer.evaluate(test_data_j, load_best_model=True, reverse=True)
    # wandb.log(test_result_j)
    logger.info('test for job result [all]: {}'.format(test_result_str_j))

    return best_valid_score, test_result_g, test_result_j, test_result_str_g, test_result_str_j


if __name__ == "__main__":
    run_id = 0

    # get config
    params = hyper_opt_preparation(model=MODEL)
    logger = getLogger()
    ori_config = params['config']

    # get all param group
    param_names = list(all_params.keys())
    params_group = itertools.product(*list(all_params.values()))   
    cur_params = dict()

    # best valid score and param
    best_valid_score = 0
    best_valid_test_str_g = ""
    best_valid_test_str_j = ""
    best_param = {}

    for param_list in params_group:
        init_seed(ori_config['seed'], ori_config['reproducibility'])
        config = deepcopy(ori_config)

        for i in range(len(param_names)):
            cur_params[param_names[i]] = param_list[i]

        run = wandb.init(
            config=cur_params,
            project='bgpjf-hyper-0202',
            name=f'BGPJF={run_id}',
            reinit=True,
            mode='disabled'
        )

        params['config'].params.update(cur_params)
        logger.info(f'RUN-ID-{run_id}')
        run_id += 1
        # print(cur_params)
        valid_score, test_result_g, test_result_j, test_result_str_g, test_result_str_j = single_run(**params)

        if valid_score > best_valid_score:
            best_valid_score = valid_score
            best_valid_test_str_g = test_result_str_g
            best_valid_test_str_j = test_result_str_j
            best_param = cur_params

    logger.info('best params: {}'.format(str(best_param)))
    logger.info('best model test for geek result [all]: {}'.format(test_result_str_g))
    logger.info('best model test for job result [all]: {}'.format(test_result_str_j))


