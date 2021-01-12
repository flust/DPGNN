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

Traditional MF Model.

```bash
python main.py -m MF
```

### MFwBERT

MF model, whose id embedding tables are initialized by pretrained BERT vectors.

```bash
python main.py -m MFwBERT
```
