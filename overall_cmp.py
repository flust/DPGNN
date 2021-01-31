import os
from collections import defaultdict

from eval import eval_preparation, eval_process


groups = ['all', 'weak', 'skilled']
metrics = ['gauc', 'p@5', 'r@5', 'mrr']
methods = ['Pop', 'MF', 'PJFNN', 'BPJFNN', 'APJFNN', 'BERT', 'MV-CoN', 'VPJF']
pth_dir = './remained'
hline = [2, 7]


model_filenames = os.listdir(pth_dir)
method2res = defaultdict(list)
for meth in methods:
    pth_file = None
    for filename in model_filenames:
        if filename[:len(meth)] == meth:
            pth_file = os.path.join(pth_dir, filename)
            break
    print(f'{meth} -> {pth_file}', flush=True)
    if pth_file is not None:
        trainer, eval_data = eval_preparation(resume_file=pth_file)
        for group in groups:
            res = eval_process(trainer, eval_data, group=group)
            for metric in metrics:
                method2res[meth].append(res[metric])

col_num = len(metrics) * len(groups)
best_vals = []
for i in range(col_num):
    best_val = -1
    for meth in method2res:
        if method2res[meth][i] > best_val:
            best_val = method2res[meth][i]
    best_vals.append(best_val)

for i, meth in enumerate(methods):
    if i in hline:
        print('\hline')
    str_lst = [meth]
    if len(method2res[meth]) == 0:
        str_lst.extend(['-'] * col_num)
    else:
        for j, val in enumerate(method2res[meth]):
            if val == best_vals[j]:
                str_lst.append('\\bm{$' + '{:.4f}'.format(val) + '$}')
            else:
                str_lst.append('${:.4f}$'.format(val))
    print(' & '.join(str_lst) + ' \\\\')

if len(methods) in hline:
    print('\hline')
