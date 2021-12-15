nohup python main.py --model=MultiGCNBERT --mutual_weight=0.5 --gpu_id=0 >./log/MultiGCNBERT_lambda_0.5.log 2>&1
nohup python main.py --model=MultiGCNBERT --mutual_weight=0.1 --gpu_id=0 >./log/MultiGCNBERT_lambda_0.1.log 2>&1
nohup python main.py --model=MultiGCNBERT --mutual_weight=0.05 --gpu_id=0 >./log/MultiGCNBERT_lambda_0.05.log 2>&1
nohup python main.py --model=MultiGCNBERT --mutual_weight=0.01 --gpu_id=0 >./log/MultiGCNBERT_lambda_0.01.log 2>&1
nohup python main.py --model=MultiGCNBERT --mutual_weight=0.005 --gpu_id=0 >./log/MultiGCNBERT_lambda_0.005.log 2>&1
nohup python main.py --model=MultiGCNBERT --mutual_weight=0.001 --gpu_id=0 >./log/MultiGCNBERT_lambda_0.001.log 2>&1