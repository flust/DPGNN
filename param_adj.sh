nohup python main.py --model=MultiGCNBERT --mutual_weight=0.5 --gpu_id=0 >./log/MultiGCNBERT_lambda_0.5.log 2>&1
nohup python main.py --model=MultiGCNBERT --mutual_weight=0.1 --gpu_id=0 >./log/MultiGCNBERT_lambda_0.1.log 2>&1
nohup python main.py --model=MultiGCNBERT --mutual_weight=0.05 --gpu_id=0 >./log/MultiGCNBERT_lambda_0.05.log 2>&1
nohup python main.py --model=MultiGCNBERT --mutual_weight=0.01 --gpu_id=0 >./log/MultiGCNBERT_lambda_0.01.log 2>&1
nohup python main.py --model=MultiGCNBERT --mutual_weight=0.005 --gpu_id=0 >./log/MultiGCNBERT_lambda_0.005.log 2>&1
nohup python main.py --model=MultiGCNBERT --mutual_weight=0.001 --gpu_id=0 >./log/MultiGCNBERT_lambda_0.001.log 2>&1



nohup python main.py --model=MultiGCNBERT --temperature=0.2 --gpu_id=1 >./log/MultiGCNBERT_temp_0.2.log 2>&1
nohup python main.py --model=MultiGCNBERT --temperature=0.1 --gpu_id=1 >./log/MultiGCNBERT_temp_0.1.log 2>&1
nohup python main.py --model=MultiGCNBERT --temperature=0.07 --gpu_id=1 >./log/MultiGCNBERT_temp_0.07.log 2>&1
nohup python main.py --model=MultiGCNBERT --temperature=0.05 --gpu_id=1 >./log/MultiGCNBERT_temp_0.05.log 2>&1
nohup python main.py --model=MultiGCNBERT --temperature=0.03 --gpu_id=1 >./log/MultiGCNBERT_temp_0.03.log 2>&1
nohup python main.py --model=MultiGCNBERT --temperature=0.01 --gpu_id=1 >./log/MultiGCNBERT_temp_0.01.log 2>&1



nohup python main.py --model=MultiGCNBERT --n_layers=1 --gpu_id=2 >./log/MultiGCNBERT_layers_1.log 2>&1
nohup python main.py --model=MultiGCNBERT --n_layers=2 --gpu_id=2 >./log/MultiGCNBERT_layers_2.log 2>&1
nohup python main.py --model=MultiGCNBERT --n_layers=3 --gpu_id=2 >./log/MultiGCNBERT_layers_3.log 2>&1
nohup python main.py --model=MultiGCNBERT --n_layers=4 --gpu_id=2 >./log/MultiGCNBERT_layers_4.log 2>&1
