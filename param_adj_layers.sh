nohup python main.py --model=MultiGCNBERT --n_layers=1 --gpu_id=2 >./log/MultiGCNBERT_layers_1.log 2>&1
nohup python main.py --model=MultiGCNBERT --n_layers=2 --gpu_id=2 >./log/MultiGCNBERT_layers_2.log 2>&1
nohup python main.py --model=MultiGCNBERT --n_layers=3 --gpu_id=2 >./log/MultiGCNBERT_layers_3.log 2>&1
nohup python main.py --model=MultiGCNBERT --n_layers=4 --gpu_id=2 >./log/MultiGCNBERT_layers_4.log 2>&1