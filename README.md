# VPJF
Codes of Person-Job Fit for Vulnerable Candidates

## Dataset

`dataset_path` in `prop/overall.yaml` should contain the following files:

```
dataset_path/
├── data.train      # interaction
├── data.test
├── data.valid
├── geek.token      # id tokens
├── job.token
├── geek.bert.npy   # pretrained bert vecs
├── job.bert.npy
├── geek.longsent   # text files after word splitting
└── job.longsent
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

MF model, whose id embedding tables are initialized by pretrained BERT vectors.

```bash
python main.py -m MFwBERT
```

Hyperparameters of reported result

```yaml
embedding_size: 756
learning_rate: 0.0003
```

### BERT

fine-tuned BERT in a double tower mode.

```bash
python main.py -m BERT
```

Hyperparameters of reported result

```yaml
embedding_size: 756
hidden_size: 64
dropout: 0.2
learning_rate: 0.001
```

## Result

| Model   | nDCG@10    | MRR@10     | MAP@10     |
| ------- | ---------- | ---------- | ---------- |
| BERT    | 0.3348     | 0.2739     | 0.2547     |
| MF      | 0.3519     | 0.2762     | **0.2840** |
| MFwBERT | **0.3529** | **0.2912** | 0.2795     |
