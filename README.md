# VPJF
Codes of Person-Job Fit for Vulnerable Candidates

## Dataset

`dataset_path` in `prop/overall.yaml` should contain the following files:

```
dataset_path/
├── data.{train/valid/test}[.{bert/his_len/job_his/qlen_his/qwd_his}.npy]
├── data.search.{train/valid/test}
├── {geek/job}.{token/sent/longsent/desc}
├── job.search.bert.npy
├── job.search.token
├── word.cnt
└── word.search.id
```

## Baseline & Method

### MF

Traditional MF Model.

### PJFNN

CNN-based Person-Job Fit Model.

> Zhu et.al. Person-Job Fit: Adapting the Right Talent for the Right Job with Joint Representation Learning. TMIS 2018.

### BPJFNN

RNN-based Person-Job Fit Model.

> Qin et.al. Enhancing Person-Job Fit for Talent Recruitment: An Ability-aware Neural Network Approach. SIGIR 2018.

### BERT

Fine-tuned BERT + MLP

### VPJFv5

* Text Matching: BERT + Linear
* Intent Modeling:
    * l_add history (64)
    * qwd emb pooling
    * job desc emb self attention
* User Modeling:
    * MF pretrained

## Usage

### Train

```bash
python main.py [-h] [--model MODEL] [--name NAME]
```

Arguments:

* `--model MODEL`, `-m MODEL` Model to test. `MODEL` should be one of:
```
MF, PJFNN, BPJFNN, BERT, VPJFv5
```
* `--name NAME`, `-n NAME` Name of this run. Defaults to `MODEL`.

### Evaluation

```bash
python eval.py [-h] [--file FILE] [--phase PHASE] [--save]
```

Arguments:
* `--file FILE`, `-f FILE`  Model file to test.
* `--phase PHASE`, `-p PHASE` Which phase to evaluate. `PHASE` should be one of `train`, `valid` and `test`. Defaults to `test`.
* `--group GROUP`, `-g GROUP` Which group to evaluate. `GROUP` should be one of `all`, `weak` and `skilled`. Defaults to `all`.
* `--save`, `-s` Whether to save predict score. Defaults to not saving.
