import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss


class Evaluator:
    def __init__(self, config):
        self.ranking_metric2func = {
            'ndcg': self._calcu_nDCG,
            'map': self._calcu_MAP,
        }
        self.topk = config['topk']
        self.maxtopk = max(self.topk)
        self.precision = config['metric_decimal_place']

        self.base = []
        self.idcg = []
        for i in range(self.maxtopk):
            self.base.append(np.log(2) / np.log(i + 2))
            if i > 0:
                self.idcg.append(self.base[i] + self.idcg[i - 1])
            else:
                self.idcg.append(self.base[i])

    def collect(self, interaction, scores):
        uid2topk = {}
        scores = scores.cpu().numpy()
        labels = interaction['label'].numpy()
        for i, uid in enumerate(interaction['geek_id'].numpy()):
            if uid not in uid2topk:
                uid2topk[uid] = []
            uid2topk[uid].append((scores[i], labels[i]))
        return uid2topk

    def evaluate(self, uid2topk_list):
        uid2topk = self._merge_uid2topk(uid2topk_list)
        result = {}
        result.update(self._calcu_ranking_metrics(uid2topk))
        result.update(self._calcu_cls_metrics(uid2topk))
        for m in result:
            result[m] = round(result[m], self.precision)
        return result

    def _calcu_ranking_metrics(self, uid2topk):
        result = {}
        for m in ['ndcg', 'map']:
            for k in self.topk:
                result[f'{m}@{k}'] = self.ranking_metric2func[m](uid2topk, k)
        result['mrr'] = self._calcu_MRR(uid2topk)
        return result

    def _calcu_cls_metrics(self, uid2topk):
        scores, labels = self._flatten_cls_list(uid2topk)
        result = {}
        result['auc'] = roc_auc_score(labels, scores)
        result['logloss'] = log_loss(labels, scores)
        return result

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
