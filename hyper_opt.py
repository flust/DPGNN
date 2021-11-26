from copy import deepcopy
from logging import getLogger
import sys

from config import Config
from data.dataset import create_datasets
from data.dataloader import construct_dataloader
from trainer import Trainer
from utils import init_seed, init_logger, dynamic_load
import itertools
import wandb


params_range = dict()
params_range['MF'] = {   
    'embedding_size': [128],
    'learning_rate': [0.001, 0.0001, 0.00001],
}

params_range['NCF'] = {
    'embedding_size': [64],  # NCF embedding_size 减半，因为 user/item 都有两个 embedding
    'mlp_hidden_size': [[32], [32, 16], [32, 16, 8]],
    'learning_rate': [0.001, 0.0001],
}

params_range['LightGCN'] = {
    'embedding_size': [128],
    'n_layers': [1, 2, 3, 4],
    'learning_rate': [0.001, 0.0001, 0.00001],
}

params_range['LightGCNa'] = {   # 增加了开聊建边的 LightGCN
    'embedding_size': [128],
    'n_layers': [2, 3],
    'learning_rate': [0.001, 0.0001, 0.00001],
}

params_range['LightGCNb'] = {   # 拆点的 LightGCN
    'embedding_size': [64],  # 因为拆点,embedding_size 减半，同NCF
    'n_layers': [2, 3],
    'learning_rate': [0.001, 0.0001, 0.00001],
}

params_range['MultiGCN'] = {
    'embedding_size': [64],   # 因为拆点,embedding_size 减半
    'n_layers': [2, 3],
    'learning_rate': [0.001, 0.0001],
}

params_range['MultiGCNs'] = {
    'embedding_size': [128],   # 因为拆点,embedding_size 减半
    'n_layers': [3],
    'learning_rate': [0.001],
}

params_range['LightGCNal'] = {   # 增加了开聊建边的 LightGCN
    'embedding_size': [128],
    'n_layers': [2, 3],
    'learning_rate': [0.001, 0.0001],
}

params_range['MultiGCNsl'] = {
    # 'embedding_size': [64],   # 因为拆点,embedding_size 减半
    'embedding_size': [128],
    'n_layers': [3],
    'omega': [10, 0.1],
    'learning_rate': [0.001],
}

params_range['MultiGCNsl1'] = {
    # 'embedding_size': [64],   # 因为拆点,embedding_size 减半
    'embedding_size': [128],
    'n_layers': [3],
    'learning_rate': [0.001],
}

params_range['MultiGCNsl1l2'] = {
    # 'embedding_size': [64],   # 因为拆点,embedding_size 减半
    'embedding_size': [128],
    'mutual_weight': [0.1, 0.01],
    'temperature': [0.1, 0.05],
    'n_layers': [3],
    'learning_rate': [0.001],
}

params_range['MultiGNN'] = {
    'embedding_size': [64],
    'n_layers': [3],
    'learning_rate': [0.001, 0.0001],
}

def get_arguments():
    args = dict()
    for arg in sys.argv[1:]:
        arg_name, arg_value = arg.split('=')
        try:
            arg_value = int(arg_value)
        except:
            try:
                arg_value = float(arg_value)
            except:
                pass
        arg_name = arg_name.strip('-')
        args[arg_name] = arg_value
    return args

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
    logger.info('test for user result [all]: {}'.format(test_result_str_g))

    # model evaluation for job
    test_result_j, test_result_str_j = trainer.evaluate(test_data_j, load_best_model=True, reverse=True)
    logger.info('test for job result [all]: {}'.format(test_result_str_j))

    return best_valid_score, test_result_g, test_result_j, test_result_str_g, test_result_str_j


if __name__ == "__main__":
    run_id = 0

    # get config
    args = get_arguments()
    MODEL = args['model'] 
    params = hyper_opt_preparation(model=MODEL)
    logger = getLogger()
    ori_config = params['config']

    # get all param group
    param_names = list(params_range[MODEL].keys())
    params_group = itertools.product(*list(params_range[MODEL].values()))   
    logger.info("All parameters range: " + str(params_range[MODEL]))

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
        logger.info(params['config'])

        valid_score, test_result_g, test_result_j, test_result_str_g, test_result_str_j = single_run(**params)

        if valid_score > best_valid_score:
            best_valid_score = valid_score
            best_valid_test_str_g = test_result_str_g
            best_valid_test_str_j = test_result_str_j
            best_param = deepcopy(cur_params)

    logger.info('best params: {}'.format(str(best_param)))
    logger.info('best model test for geek result [all]: {}'.format(best_valid_test_str_g))
    logger.info('best model test for job result [all]: {}'.format(best_valid_test_str_j))


