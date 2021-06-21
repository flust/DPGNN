
# nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.01 -do=0.2 -nl=2 >./log/LightGCN_lr01_n2.log 2>&1 
# echo "LightGCN lr1 end"
# nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.003 -do=0.2 -nl=2  >./log/LightGCN_lr003_n2.log 2>&1 
# echo "LightGCN lr2 end"
# nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.001 -do=0.2 -nl=2  >./log/LightGCN_lr001_n2.log 2>&1 
# echo "LightGCN lr3 end"
# nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.0003 -do=0.2 -nl=2  >./log/LightGCN_lr0003_n2.log 2>&1 
# echo "LightGCN lr4 end"
# nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.0001 -do=0.2 -nl=2  >./log/LightGCN_lr0001_n2.log 2>&1 
# echo "LightGCN lr5 end"
# nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.00001 -do=0.2 -nl=2  >./log/LightGCN_lr00001_n2.log 2>&1 
# echo "LightGCN lr6 end"
# nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.00003 -do=0.2 -nl=2  >./log/LightGCN_lr00003_n2.log 2>&1 
# echo "LightGCN lr7 end"
# nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.000005 -do=0.2 -nl=2  >./log/LightGCN_lr000005_n2.log 2>&1  
# echo "LightGCN lr8 end"

# lr 0.0001
nohup python main.py -m=LightGCN -d=multi -es=128 -lr=0.0001 -do=0.2 -nl=2 >./log/LightGCN_es128_n2.log 2>&1 
echo "LightGCN es1 end"
nohup python main.py -m=LightGCN -d=multi -es=256 -lr=0.0001 -do=0.2 -nl=2  >./log/LightGCN_es256_n2.log 2>&1 
echo "LightGCN es2 end"
nohup python main.py -m=LightGCN -d=multi -es=512 -lr=0.0001 -do=0.2 -nl=2  >./log/LightGCN_es512_n2.log 2>&1 
echo "LightGCN es3 end"
nohup python main.py -m=LightGCN -d=multi -es=768 -lr=0.0001 -do=0.2 -nl=2  >./log/LightGCN_es768_n2.log 2>&1 
echo "LightGCN es4 end"
nohup python main.py -m=LightGCN -d=multi -es=1024 -lr=0.0001 -do=0.2 -nl=2  >./log/LightGCN_es1024_n2.log 2>&1 
echo "LightGCN es5 end"
