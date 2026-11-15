"""Does forgetting hurt search? Re-solve the 400 training tasks (budget 250, no retrieval) with the policy after a 5-chunk stream."""
import os
os.environ['ARC_DSL'] = 'v2'; os.environ['ARC_RES'] = '/home/claude/results_v2'
import json, random, numpy as np
from multiprocessing import Pool
import run_all as RA
from arcpipe import *
dec = json.load(open(RA.R + '/decisions.json')); kw = dict(lam=dec['lam'], depth=dec['depth'])
r0 = RA.load('train_r0'); r1 = RA.load('train_r1'); mem0 = RA.mem_from(r0, RA.FT); mem1 = RA.mem_from(r1, RA.FT)
for n, m in mem0.items(): mem1.setdefault(n, m)
d1 = RA.data_from(mem1); pi2 = Policy(RA.MU, RA.SD); pi2.fit(d1)
E = RA.load('eval_v2_full'); items = {n: (RA.FE[n], tuple(m['progs'][0])) for n, m in RA.mem_from(E, RA.FE).items() if m['progs'][0]}
names = sorted(RA.ev)
def stream(mode, seed=0):
    rng = random.Random(seed); order = names[:]; rng.shuffle(order); chunks = [order[i::5] for i in range(5)]
    pol = pi2.clone(); pol.F = None; pol.anchor = None
    if mode == 'ewc': pol.consolidate(d1, lam=100.0)
    for ch in chunks:
        nd = [items[n] for n in ch if n in items]
        if not nd: continue
        if mode == 'naive': pol.fit(nd, steps=300, lr=0.3)
        elif mode == 'replay': pol.fit(nd + d1, steps=300, lr=0.3)
        else: pol.fit(nd, steps=300, lr=0.3, ewc=True); pol.consolidate(nd, lam=100.0)
    return pol
if __name__ == '__main__':
    pool = Pool(2); out = {}
    for mode in ['frozen', 'naive', 'ewc', 'replay']:
        pol = pi2 if mode == 'frozen' else stream(mode)
        r = RA.run_config(f'ret_{mode}', RA.tr, RA.FT, pol, None, budget=250, pool=pool, **kw)
        out[mode] = {'solved': sum(v['solved'] for k, v in r.items() if k != '_meta'), 'nll_s1': float(pol.nll(d1))}
        print(mode, out[mode], flush=True)
    json.dump(out, open(RA.R + '/retention.json', 'w')); print('DONE', flush=True)
