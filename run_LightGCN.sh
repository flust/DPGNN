
nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.01 -do=0.2 >./log/LightGCN_lr01.log 2>&1 
echo "LightGCN lr1 end"
nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.003 -do=0.2 >./log/LightGCN_lr003.log 2>&1 
echo "LightGCN lr2 end"
nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.001 -do=0.2 >./log/LightGCN_lr001.log 2>&1 
echo "LightGCN lr3 end"
nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.0003 -do=0.2 >./log/LightGCN_lr0003.log 2>&1 
echo "LightGCN lr4 end"
nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.0001 -do=0.2 >./log/LightGCN_lr0001.log 2>&1 
echo "LightGCN lr5 end"
nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.00001 -do=0.2 >./log/LightGCN_lr00001.log 2>&1 
echo "LightGCN lr6 end"
nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.00003 -do=0.2 >./log/LightGCN_lr00003.log 2>&1 
echo "LightGCN lr7 end"
nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.000005 -do=0.2 >./log/LightGCN_lr000005.log 2>&1  
echo "LightGCN lr8 end"