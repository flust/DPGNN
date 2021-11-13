# nohup python main.py --model=MF >./log/MF.log 2>&1 &
# nohup python main.py --model=NCF >./log/NCF.log 2>&1 &
# nohup python main.py --model=LightGCN >./log/LightGCN.log 2>&1 &
nohup python main.py --model=LightGCN2 -learning_rate=0.001 >./log/LightGCN2.log 2>&1 &
nohup python main.py --model=LightGCN2 -learning_rate=0.00001 >./log/LightGCN2.log 2>&1 &
