import torch
import torch.nn as nn

from utils import init_seed, dynamic_load
from tqdm import tqdm

'''
Hyperparameters
'''
topk = 20
train_data_path = 'dataset/bosszp/data.train'
eval_data_path = 'dataset/bosszp/data.test'
output_score_path = 'ItemKNN.score'
default_score = 0.5


'''
Main Process
'''
def load_train():
    geek2job = {}
    with open(train_data_path, 'r', encoding='utf-8') as file:
        for line in file:
            geek_id, job_id, ts, label = line.strip().split('\t')
            if geek_id not in geek2job:
                geek2job[geek_id] = {}
            geek2job[geek_id][job_id] = int(label)
    return geek2job


checkpoint = torch.load('remained/MF-Jan-13-2021_16-28-28.pth')

config = checkpoint['config']
init_seed(config['seed'], config['reproducibility'])

# data preparation
pool = dynamic_load(config, 'data.pool', 'Pool')(config)

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
job_emb = nn.Embedding(pool.job_num, pool.embedding_size)

geek2job = load_train()

with torch.no_grad(), open(output_score_path, 'w', encoding='utf-8') as score_file:
    with open(eval_data_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            geek_id, job_id, ts, label = line.strip().split('\t')
            lst = []
            job_id = pool.job_token2id[job_id]
            job_vec = job_emb(torch.LongTensor([job_id]))
            if geek_id not in geek2job:
                score_file.write(f'{default_score}\n')
                continue
            for item, lb in geek2job[geek_id].items():
                item_id = pool.job_token2id[item]
                item_vec = job_emb(torch.LongTensor([item_id]))
                score = cos(job_vec, item_vec).numpy()[0]
                lst.append((score, lb))
            lst.sort(key=lambda t: t[0], reverse=True)
            cnt = 0
            tot = 0
            for i, (score, label) in enumerate(lst):
                if i >= topk:
                    break
                cnt += label * score
                tot += score
            score_file.write(f'{cnt / tot}\n')
