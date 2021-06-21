
# nohup python main.py -m=MultiPJF -d=multi -es=128 -lr=0.01 -do=0.2 >./log/BGPJF_lr01.log 2>&1 
# echo "MultiPJF lr1 end"
# nohup python main.py -m=MultiPJF -d=multi -es=128 -lr=0.003 -do=0.2 >./log/BGPJF_lr003.log 2>&1 
# echo "MultiPJF lr2 end"
# nohup python main.py -m=MultiPJF -d=multi -es=128 -lr=0.001 -do=0.2 >./log/BGPJF_lr001.log 2>&1 
# echo "MultiPJF lr3 end"
# nohup python main.py -m=MultiPJF -d=multi -es=128 -lr=0.0003 -do=0.2 >./log/BGPJF_lr0003.log 2>&1 
# echo "MultiPJF lr4 end"
# nohup python main.py -m=MultiPJF -d=multi -es=128 -lr=0.0001 -do=0.2 >./log/BGPJF_lr0001.log 2>&1 
# echo "MultiPJF lr5 end"
# nohup python main.py -m=MultiPJF -d=multi -es=128 -lr=0.00001 -do=0.2 >./log/BGPJF_lr00001.log 2>&1
# echo "MultiPJF lr6 end"
# nohup python main.py -m=MultiPJF -d=multi -es=128 -lr=0.00003 -do=0.2 >./log/BGPJF_lr00003.log 2>&1
# echo "MultiPJF lr7 end"
# nohup python main.py -m=MultiPJF -d=multi -es=128 -lr=0.000005 -do=0.2 >./log/BGPJF_lr000005.log 2>&1
# echo "MultiPJF lr8 end"
# lr 0.00001 / 0.000005 / 0.0001 差不多
# lr 0.00001 验证集最好

# 下面调 embedding size
# nohup python main.py -m=MultiPJF -d=multi -es=16 -lr=0.00001 -do=0.2 >./log/BGPJF_es16.log 2>&1 
# echo "MultiPJF es1 end"
# nohup python main.py -m=MultiPJF -d=multi -es=32 -lr=0.00001 -do=0.2 >./log/BGPJF_es32.log 2>&1 
# echo "MultiPJF es2 end"
# nohup python main.py -m=MultiPJF -d=multi -es=64 -lr=0.00001 -do=0.2 >./log/BGPJF_es64.log 2>&1 
# echo "MultiPJF es3 end"
# nohup python main.py -m=MultiPJF -d=multi -es=100 -lr=0.00001 -do=0.2 >./log/BGPJF_es100.log 2>&1 
# echo "MultiPJF es4 end"
# nohup python main.py -m=MultiPJF -d=multi -es=128 -lr=0.00001 -do=0.2 >./log/BGPJF_es128.log 2>&1 
# echo "MultiPJF es5 end"
# nohup python main.py -m=MultiPJF -d=multi -es=150 -lr=0.00001 -do=0.2 >./log/BGPJF_es150.log 2>&1
# echo "MultiPJF es6 end"
# nohup python main.py -m=MultiPJF -d=multi -es=200 -lr=0.00001 -do=0.2 >./log/BGPJF_es200.log 2>&1
# echo "MultiPJF es7 end"
# nohup python main.py -m=MultiPJF -d=multi -es=256 -lr=0.00001 -do=0.2 >./log/BGPJF_es256.log 2>&1
# echo "MultiPJF es8 end"

# embedding size 256


# # 下面调层数
nohup python main.py -m=MultiPJF -d=multi -es=256 -lr=0.00001 -do=0.2 >./log/BGPJF_es256.log 2>&1
echo "MultiPJF es8 end"