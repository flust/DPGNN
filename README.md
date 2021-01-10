# VPJF
Codes of Person-Job Fit for Vulnerable Candidates

## Dataset

A dir named `dataset` should exist where `main.py` lies, containing:

```
dataset/
├── data.train      # interaction
├── data.test
├── data.valid
├── geek.token      # id tokens
├── job.token
├── geek.bert.npy   # pretrained bert vecs
└── job.bert.npy
```

## Baseline

### MF

```bash
python main.py -m MF
```

Hyperparameters of reported result

```yaml
embedding_size: 64
learning_rate: 0.001
```

## Result

| Model | nDCG@10 | MRR@10 | MAP@10 |
| ----- | ------- | ------ | ------ |
| MF    | 0.2035  | 0.1523 | 0.1577 |
