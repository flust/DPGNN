
# nohup python main.py -m=MF -d=multi -es=128 -lr=0.01 -do=0.2 >./log/MF_lr01.log 2>&1 
# echo "MF lr1 end"
# nohup python main.py -m=MF -d=multi -es=128 -lr=0.003 -do=0.2 >./log/MF_lr003.log 2>&1 
# echo "MF lr2 end"
# nohup python main.py -m=MF -d=multi -es=128 -lr=0.001 -do=0.2 >./log/MF_lr001.log 2>&1 
# echo "MF lr3 end"
# nohup python main.py -m=MF -d=multi -es=128 -lr=0.0003 -do=0.2 >./log/MF_lr0003.log 2>&1 
# echo "MF lr4 end"
# nohup python main.py -m=MF -d=multi -es=128 -lr=0.0001 -do=0.2 >./log/MF_lr0001.log 2>&1 
# echo "MF lr5 end"
# nohup python main.py -m=MF -d=multi -es=128 -lr=0.00001 -do=0.2 >./log/MF_lr00001.log 2>&1 & 
# # echo "MF lr1 end"
# nohup python main.py -m=MF -d=multi -es=128 -lr=0.00003 -do=0.2 >./log/MF_lr00003.log 2>&1 &
# # echo "MF lr2 end"
# nohup python main.py -m=MF -d=multi -es=128 -lr=0.000005 -do=0.2 >./log/MF_lr000005.log 2>&1 & 
# echo "MF lr1 end"
# lr 0.00001 / 0.00003 差不多
# lr 0.00001

# 下面调 embedding size
# nohup python main.py -m=MF -d=multi -es=128 -lr=0.00001 -do=0.2 >./log/MF_es128.log 2>&1
# echo "MF lr2 end"
# nohup python main.py -m=MF -d=multi -es=64 -lr=0.00001 -do=0.2 >./log/MF_es64.log 2>&1
# echo "MF lr3 end"
# nohup python main.py -m=MF -d=multi -es=256 -lr=0.00001 -do=0.2 >./log/MF_es256.log 2>&1
# echo "MF lr4 end"
# nohup python main.py -m=MF -d=multi -es=100 -lr=0.00001 -do=0.2 >./log/MF_es100.log 2>&1
# echo "MF lr5 end"
# nohup python main.py -m=MF -d=multi -es=200 -lr=0.00001 -do=0.2 >./log/MF_es200.log 2>&1
# echo "MF lr6 end"
# nohup python main.py -m=MF -d=multi -es=32 -lr=0.00001 -do=0.2 >./log/MF_es32.log 2>&1
# echo "MF lr7 end"

# embedding size 64


# 下面调dropout
nohup python main.py -m=MF -d=multi -es=64 -lr=0.00001 -do=0.1 >./log/MF_do1.log 2>&1
echo "MF do1 end"
nohup python main.py -m=MF -d=multi -es=64 -lr=0.00001 -do=0.3 >./log/MF_do3.log 2>&1
echo "MF do3 end"
nohup python main.py -m=MF -d=multi -es=64 -lr=0.00001 -do=0.4 >./log/MF_do4.log 2>&1
echo "MF do4 end"