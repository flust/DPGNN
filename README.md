# DPGNN
Code for RecSys 2022 submission:
> Modeling Two-Way Selection Preference for Person-Job Fit

## Requirements
```
torch==1.10.0+cu113
torch_geometric==2.0.2
cudatoolkit==11.3.1
```

## Dataset

`dataset_path` in `prop/overall.yaml` should contain the following files:

```
dataset_path/
├── data.{train/valid_g/valid_j/test_g/test_j/user_add/job_add}
├── {geek/job}.bert.npy
└── {geek/job}.token
```

**Our dataset will be anonymized and released after the reviewing period.**

### Train

```bash
python main.py
```