# VPJF
Codes of Person-Job Fit for Vulnerable Candidates

## Dataset

`dataset_path` in `prop/overall.yaml` should contain the following files:

```
dataset_path/
├── data.train              # interaction
├── data.test
├── data.valid
├── geek.token              # id tokens
├── job.token
├── geek.longsent           # text files after word splitting
├── job.longsent
├── data.train.bert.npy     # pretrained bert vecs
├── data.test.bert.npy
├── data.valid.bert.npy
├── geek.bert.npy
└── job.bert.npy
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

### BPJFNN

Text-based Person-Job Fit Model.

> Chuan Qin et.al. Enhancing Person-Job Fit for Talent Recruitment: An Ability-aware Neural Network Approach. SIGIR 2018.

```bash
python main.py -m BPJFNN
```

### BERT

Fine-tuned BERT + MLP

```bash
python main.py -m BERT
```

### VPJFv1

* Text Matching: BERT
* Intent Modeling:
    * l_add history (64)
    * qwd emb pooling
    * job desc emb pooling
