# nohup python main.py --model=MF --direction=multi >MF128.log 2>&1 
# echo "MF end"
# nohup python main.py --model=LightGCN --direction=multi >LightGCN128.log 2>&1 
# echo "LightGCN end"
# nohup python main.py --model=MultiGCN --direction=multi >MultiGCN128.log 2>&1 
# echo "MultiGCN end"
nohup python main.py --model=MultiPJF --direction=multi >MultiPJF64.log 2>&1 
echo "MultiPJF end"
nohup python main.py --model=PJFNN --direction=multi >PJFNN.log 2>&1 
echo "PJFNN end"
nohup python main.py --model=BPJFNN --direction=multi >BPJFNN.log 2>&1 
echo "BPJFNN end"