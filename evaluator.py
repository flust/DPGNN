import numpy as np


class Evaluator:
    def __init__(self, config):
        self.metric2func = {
            'ndcg': self._calcu_nDCG,
            'mrr': self._calcu_MRR,
            'map': self._calcu_MAP,

        }
        self.metrics = [_.lower() for _ in config['metrics']]
        self.topk = config['topk']
        self.precision = config['metric_decimal_place']

        self.base = []
        self.idcg = []
        for i in range(self.topk):
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
        return self._sort_cut_uid2topk(uid2topk)

    def evaluate(self, uid2topk_list, eval_data):
        uid2topk = self._merge_uid2topk(uid2topk_list)
        result = {}
        for m in self.metrics:
            result[f'{m}@{self.topk}'] = round(self.metric2func[m](uid2topk), self.precision)
        return result

    def _calcu_nDCG(self, uid2topk):
        tot = 0
        for uid in uid2topk:
            dcg = 0
            pos = 0
            for i, (score, lb) in enumerate(uid2topk[uid]):
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

    def _calcu_MAP(self, uid2topk):
        tot = 0
        for uid in uid2topk:
            tp = 0
            pos = 0
            ap = 0
            for i, (score, lb) in enumerate(uid2topk[uid]):
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
        return self._sort_cut_uid2topk(uid2topk)

    def _sort_cut_uid2topk(self, uid2topk):
        for uid in uid2topk:
            uid2topk[uid].sort(key=lambda t: t[0], reverse=True)
            uid2topk[uid] = uid2topk[uid][:self.topk]
        return uid2topk
