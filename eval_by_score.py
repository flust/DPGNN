import os
import argparse
import numpy as np
from evaluator import Evaluator


params = {
    'dataset_path': './dataset/bosszp0203/',
    'topk': [5],
    'metric_decimal_place': 4,
    'metrics': ['gauc', 'p@5', 'r@5', 'mrr']
}


token2id = {}
filepath = os.path.join(params['dataset_path'], f'geek.token')
print(f'Loading {filepath}', flush=True)
with open(filepath, 'r') as file:
    for i, line in enumerate(file):
        token = line.strip()
        token2id[token] = i


evaluator = Evaluator(params)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, help='Model to eval.')
    parser.add_argument('--phase', '-p', type=str, default='test', help='Which phase to evaluate.')

    args = parser.parse_args()
    return args


def load_score(model, phase):
    score_lst = []
    with open(f'{phase}_score/{model}.score', 'r') as file:
        for line in file:
            score_lst.append(float(line.strip()))
    scores = np.array(score_lst)
    return scores


def evaluate_model(scores, phase, group='all'):
    gid2topk = {}
    with open(os.path.join(params['dataset_path'], f'data.{phase}'), 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            gid, jid, ts, label = line.strip().split('\t')
            gid = token2id[gid]
            score = float(scores[i])
            if gid not in gid2topk:
                gid2topk[gid] = []
            gid2topk[gid].append((score, int(label)))

    return evaluator.evaluate([gid2topk], group=group)


def eval_all(model, phase='test'):
    scores = load_score(model, phase)
    res, res_str = evaluate_model(scores, phase, group='all')
    print(res_str)
    res, res_str = evaluate_model(scores, phase, group='weak')
    print(res_str)
    res, res_str = evaluate_model(scores, phase, group='skilled')
    print(res_str)


if __name__ == "__main__":
    args = get_arguments()
    eval_all(args.model, args.phase)
