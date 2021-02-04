import os


dataset_path = 'dataset/bosszp0203'


# Job Post
job_token = []
for target in ['token', 'search.token']:
    with open(os.path.join(dataset_path, f'job.{target}'), 'r') as file:
        for line in file:
            job_token.append(line.strip())
print(f'        \#Job Posts & $-$ & $-$ & ${len(job_token) - 1}$ \\\\')

# Candidates
geek_tokens = []
geek2weak = {}
V_cnt = 0
S_cnt = 0
with open(os.path.join(dataset_path, 'geek.token'), 'r') as token_file, \
     open(os.path.join(dataset_path, 'geek.weak'), 'r') as weak_file:
    for i, line in enumerate(token_file):
        gid = line.strip()
        geek_tokens.append(gid)
        gid2, is_weak = weak_file.readline().strip().split('\t')
        assert gid == gid2
        is_weak = int(is_weak)
        geek2weak[gid] = is_weak
        if is_weak:
            V_cnt += 1
        else:
            S_cnt += 1
print(f'        \#Candidates & ${V_cnt}$ & ${S_cnt}$ & ${len(geek_tokens) - 1}$ \\\\')

# Accept
vpos = vneg = spos = sneg = 0
for target in ['train', 'test', 'valid']:
    with open(os.path.join(dataset_path, f'data.{target}'), 'r') as file:
        for line in file:
            gid, jid, ts, label = line.strip().split('\t')
            if geek2weak[gid]:
                if label == '1':
                    vpos += 1
                else:
                    vneg += 1
            else:
                if label == '1':
                    spos += 1
                else:
                    sneg += 1
print(f'        \#Accept & ${vpos}$ & ${spos}$ & ${vpos + spos}$ \\\\')
print(f'        \#Browsing & ${vneg}$ & ${sneg}$ & ${vneg + sneg}$ \\\\')
tot = vpos + vneg + spos + sneg

# Search History
vhislen = shislen = vqwd_len = sqwd_len = 0
for target in ['train', 'test', 'valid']:
    with open(os.path.join(dataset_path, f'data.search.{target}'), 'r', encoding='utf-8') as file:
        for line in file:
            gid, jid, label, job_his, qwd_his, qwd_len = line.strip().split('\t')
            qwd_len = qwd_len.split(' ')
            if geek2weak[gid]:
                vhislen += len(qwd_len)
                for _ in qwd_len:
                    vqwd_len += int(_)
            else:
                shislen += len(qwd_len)
                for _ in qwd_len:
                    sqwd_len += int(_)
print('        $\overline{|q|}$ & $', vhislen / (vpos + vneg), f'$ & ${shislen / (spos + sneg)}$ & ${(shislen + vhislen) / tot}$ \\\\')
print('        $\overline{|\mathcal{H}|}$ & $', vqwd_len / vhislen, f'$ & ${sqwd_len / shislen}$ & ${(vqwd_len + sqwd_len) / (vhislen + shislen)}$ \\\\')

