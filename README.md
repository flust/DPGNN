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
embedding_size: 16
learning_rate: 0.0001
```

### MFwBERT

```bash
python main.py -m MFwBERT
```

Hyperparameters of reported result

```yaml
embedding_size: 756
learning_rate: 0.0003
```

## Result

| Model   | nDCG@10    | MRR@10     | MAP@10     |
| ------- | ---------- | ---------- | ---------- |
| MF      | 0.3519     | 0.2762     | **0.2840** |
| MFwBERT | **0.3529** | **0.2912** | 0.2795     |
