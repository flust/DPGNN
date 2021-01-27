import os
from logging import getLogger

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss


class Evaluator:
    def __init__(self, config):
        self.logger = getLogger()
        self.ranking_metric2func = {
            'ndcg': self._calcu_nDCG,
            'map': self._calcu_MAP,
        }
        self.topk = config['topk']
        self.maxtopk = max(self.topk)
        self.precision = config['metric_decimal_place']
        if config['metrics'] is not None:
            self.metrics = config['metrics']
        else:
            self.metrics = ['auc', 'gauc', 'map@5', 'map@10', 'mrr']

        self.base = []
        self.idcg = []
        for i in range(self.maxtopk):
            self.base.append(np.log(2) / np.log(i + 2))
            if i > 0:
                self.idcg.append(self.base[i] + self.idcg[i - 1])
            else:
                self.idcg.append(self.base[i])

        self._load_geek2weak(config['dataset_path'])

    def collect(self, interaction, scores):
        uid2topk = {}
        scores = scores.cpu().numpy()
        labels = interaction['label'].numpy()
        for i, uid in enumerate(interaction['geek_id'].numpy()):
            if uid not in uid2topk:
                uid2topk[uid] = []
            uid2topk[uid].append((scores[i], labels[i]))
        return uid2topk

    def evaluate(self, uid2topk_list, group='all'):
        uid2topk = self._merge_uid2topk(uid2topk_list)
        uid2topk = self._filter_illegal(uid2topk)
        uid2topk = self._filter_group(uid2topk, group)
        result = {}
        result.update(self._calcu_ranking_metrics(uid2topk))
        result.update(self._calcu_cls_metrics(uid2topk))
        for m in result:
            result[m] = round(result[m], self.precision)
        return result, self._format_str(result)

    def _format_str(self, result):
        res = ''
        for metric in self.metrics:
            res += '\n\t{}:\t{:.4f}'.format(metric, result[metric])
        return res

    def _calcu_ranking_metrics(self, uid2topk):
        result = {}
        for m in ['ndcg', 'map']:
            for k in self.topk:
                metric = f'{m}@{k}'
                if metric in self.metrics:
                    result[metric] = self.ranking_metric2func[m](uid2topk, k)
        if 'mrr' in self.metrics:
            result['mrr'] = self._calcu_MRR(uid2topk)
        return result

    def _calcu_cls_metrics(self, uid2topk):
        scores, labels = self._flatten_cls_list(uid2topk)
        result = {}
        result['auc'] = roc_auc_score(labels, scores)
        result['logloss'] = log_loss(labels, scores)
        result['gauc'] = self._calcu_GAUC(uid2topk)
        return result

    def _calcu_GAUC(self, uid2topk):
        weight_sum = auc_sum = 0
        for uid, lst in uid2topk.items():
            score_list, lb_list = zip(*lst)
            scores = np.array(score_list)
            labels = np.array(lb_list)
            w = len(labels)
            auc = roc_auc_score(labels, scores)
            weight_sum += w
            auc_sum += auc * w
        return float(auc_sum / weight_sum)

    def _calcu_nDCG(self, uid2topk, k):
        tot = 0
        for uid in uid2topk:
            dcg = 0
            pos = 0
            for i, (score, lb) in enumerate(uid2topk[uid][:k]):
                dcg += lb * self.base[i]
                pos += lb
            tot += dcg / self.idcg[int(pos) - 1]
        return tot / len(uid2topk)

    def _calcu_MRR(self, uid2topk):
        tot = 0
        for uid in uid2topk:
            for i, (score, lb) in enumerate(uid2topk[uid]):
                if lb == 1:
                    tot += 1 / (i + 1)
                    break
        return tot / len(uid2topk)

    def _calcu_MAP(self, uid2topk, k):
        tot = 0
        for uid in uid2topk:
            tp = 0
            pos = 0
            ap = 0
            for i, (score, lb) in enumerate(uid2topk[uid][:k]):
                if lb == 1:
                    tp += 1
                    pos += 1
                    ap += tp / (i + 1)
            if pos > 0:
                tot += ap / pos
        return tot / len(uid2topk)

    def _merge_uid2topk(self, uid2topk_list):
        uid2topk = {}
        for single_uid2topk in uid2topk_list:
            for uid in single_uid2topk:
                if uid not in uid2topk:
                    uid2topk[uid] = []
                uid2topk[uid].extend(single_uid2topk[uid])
        return self._sort_uid2topk(uid2topk)

    def _load_geek2weak(self, dataset_path):
        self.geek2weak = []
        filepath = os.path.join(dataset_path, f'geek.weak')
        self.logger.info(f'Loading {filepath}')
        with open(filepath, 'r') as file:
            for line in file:
                token, weak = line.strip().split('\t')
                self.geek2weak.append(int(weak))

    def _filter_illegal(self, uid2topk):
        new_uid2topk = {}
        for uid, lst in uid2topk.items():
            score_list, lb_list = zip(*lst)
            lb_sum = sum(lb_list)
            if lb_sum == 0 or lb_sum == len(lb_list):
                continue
            new_uid2topk[uid] = uid2topk[uid]
        return new_uid2topk

    def _filter_group(self, uid2topk, group):
        if group == 'all':
            return uid2topk
        elif group in ['weak', 'skilled']:
            self.logger.info(f'Evaluating on [{group}]')
            flag = 1 if group == 'weak' else 0
            new_uid2topk = {}
            for uid in uid2topk:
                if abs(self.geek2weak[uid] - flag) < 0.1:
                    new_uid2topk[uid] = uid2topk[uid]
            return new_uid2topk
        else:
            raise NotImplementedError(f'Not support [{group}]')

    def _sort_uid2topk(self, uid2topk):
        for uid in uid2topk:
            uid2topk[uid].sort(key=lambda t: t[0], reverse=True)
        return uid2topk

    def _flatten_cls_list(self, uid2topk):
        scores = []
        labels = []
        for uid, lst in uid2topk.items():
            score_list, lb_list = zip(*lst)
            scores.append(np.array(score_list))
            labels.append(np.array(lb_list))
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        assert scores.shape[0] == labels.shape[0]
        return scores, labels
