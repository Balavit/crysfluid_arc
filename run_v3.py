import os, sys
os.environ['ARC_DSL'] = 'v2'; os.environ['ARC_RES'] = '/home/claude/results_v3'
gate = sys.argv[1] if len(sys.argv) > 1 else '1'
os.environ['ARC_INDUCE_GATE'] = gate
import json, numpy as np
from multiprocessing import Pool
import run_all as RA
from arcpipe import *
cnt = lambda r: sum(v['solved'] for k, v in r.items() if k != '_meta')
ver = lambda r: sum(v['verified'] for k, v in r.items() if k != '_meta')
if __name__ == '__main__':
    pool = Pool(2); rc = RA.run_config; sfx = '' if gate == '1' else '_nogate'
    if gate == '1':
        a = rc('train_v3_noind_b250', RA.tr, RA.FT, None, pool=pool, budget=250, lam=2.0, depth=3, induce=False)
    b = rc(f'train_v3_ind_b250{sfx}', RA.tr, RA.FT, None, pool=pool, budget=250, lam=2.0, depth=3, induce=True)
    rc(f'eval_v3_ind_b250{sfx}', RA.ev, RA.FE, None, pool=pool, budget=250, lam=2.0, depth=3, induce=True)
    if gate == '1':
        os.environ['ARC_RES'] = '/home/claude/results_v2'
        R2 = '/home/claude/results_v2/'
        r0 = json.load(open(R2 + 'train_r0.json')); r1 = json.load(open(R2 + 'train_r1.json'))
        mem0 = RA.mem_from(r0, RA.FT); mem1 = RA.mem_from(r1, RA.FT)
        for n, m in mem0.items(): mem1.setdefault(n, m)
        d1 = RA.data_from(mem1); pi2 = Policy(RA.MU, RA.SD); pi2.fit(d1)
        RA.R = '/home/claude/results_v3'
        rc('eval_v3_full_b500', RA.ev, RA.FE, pi2, mem1, pool=pool, budget=500, lam=2.0, depth=3, induce=True)
    print('DONE', flush=True)
