# DPGNN
Codes of Modeling Two-Way Selection Preference for Person-Job Fit

<!-- ## Dataset

`dataset_path` in `prop/overall.yaml` should contain the following files:

```
dataset_path/
├── data.{train/valid_g/valid_j/test_g/test_j/user_add/job_add}[.{bert.npy]
├── {geek/job}.{token/sent/longsent}
└── word.cnt
```

## Baseline & Method

### Pop

Scores are calculated by users'popularity and jobs' popularity.

### MF

Traditional MF Model.

### LightGCN

> LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

### LFRR

> Neve et.al. Latent factor models and aggregation operators for collaborative filtering in reciprocal recommender systems

### PJFNN

CNN-based Person-Job Fit Model.

> Zhu et.al. Person-Job Fit: Adapting the Right Talent for the Right Job with Joint Representation Learning. TMIS 2018.

### BPJFNN

RNN-based Person-Job Fit Model.

> Qin et.al. Enhancing Person-Job Fit for Talent Recruitment: An Ability-aware Neural Network Approach. SIGIR 2018.

### BERT

Fine-tuned BERT + MLP

### IPJF

> RanLe et.al. Towards Effective and Interpretable Person-Job Fitting

### PJFFF

> Jiang et.al. Learning Effective Representations for Person-Job Fit by Feature Fusion

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
* `--phase PHASE`, `-p PHASE` Which phase to evaluate. `PHASE` should be one of `train`, `valid_g`, `valid_j`, `test_g` and `test_j`. Defaults to `test_g`.
* `--save`, `-s` Whether to save predict score. Defaults to not saving. -->
