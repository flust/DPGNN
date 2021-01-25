import numpy as np
from evaluator import Evaluator
import wandb


ensembled_methods = ['MF', 'MF', 'BERT']
params = {
    'topk': [1, 5, 10],
    'metric_decimal_place': 4,
    'metrics': ['auc', 'map@5', 'map@10', 'mrr']
}

evaluator = Evaluator(params)


run = wandb.init(
    config=params,
    project='vpjf',
    name='ensemble',
    mode='disabled'
)


def load_ensembled_methods(ensembled_methods, phase):
    scores = []
    for meth in ensembled_methods:
        score_lst = []
        with open(f'{phase}_score/{meth}.score', 'r') as file:
            for line in file:
                score_lst.append(float(line.strip()))
        score_lst = np.array(score_lst)
        scores.append(score_lst)
    return scores


def evaluate_ensembled_methods(scores, weights, phase):
    weights = np.array(weights) / sum(weights)
    new_scores = []
    for w, s in zip(weights, scores):
        new_scores.append(w * s)
    pre_score = np.sum(new_scores, axis=0)

    gid2topk = {}
    with open(f'dataset/bosszp/data.{phase}', 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            gid, jid, ts, label = line.strip().split('\t')
            score = float(pre_score[i])
            if gid not in gid2topk:
                gid2topk[gid] = []
            gid2topk[gid].append((score, int(label)))

    return evaluator.evaluate([gid2topk])


def train():
    scores = load_ensembled_methods(ensembled_methods, 'valid')

    best_metric = 0
    best_weight = None
    best_res = None

    for wa in np.arange(0, 1.1, 0.1):
        # wb = 0
        for wb in np.arange(0, 1.1, 0.1):
            wc = 1 - wa - wb
            if wc < 0:
                continue
            weights = [
                round(float(wa), 1),
                round(float(wb), 1),
                round(float(wc), 1)
            ]
            tmp_res, tmp_res_str = evaluate_ensembled_methods(scores, weights, 'valid')
            if tmp_res['auc'] + tmp_res['map@5'] > best_metric:
                best_metric = tmp_res['auc'] + tmp_res['map@5']
                best_weight = weights
                best_res = tmp_res
                wandb.log(best_res)

                print(weights, tmp_res_str)
    return best_weight


def test(weights):
    scores = load_ensembled_methods(ensembled_methods, 'test')
    res, res_str = evaluate_ensembled_methods(scores, weights, 'test')
    wandb.log(res)
    return res_str


if __name__ == "__main__":
    best_weights = train()
    print(test(best_weights))
