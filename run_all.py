import json, os, sys, time, random
import numpy as np
from multiprocessing import Pool
from arcpipe import *

BUDGET = 1000
R = os.environ.get('ARC_RES', '/home/claude/results'); os.makedirs(R, exist_ok=True)
def save(n, o): json.dump(o, open(f'{R}/{n}.json', 'w'))
def load(n):
    p = f'{R}/{n}.json'
    return json.load(open(p)) if os.path.exists(p) else None

tr = load_tasks('training'); ev = load_tasks('evaluation')
FT = {k: features(v) for k, v in tr.items()}; FE = {k: features(v) for k, v in ev.items()}
allf = np.array(list(FT.values())); MU = allf.mean(0); SD = allf.std(0) + 1e-6   # standardiser from TRAINING tasks only

def retrieve(f, memory, k=5, exclude=None):
    items = [(n, m) for n, m in memory.items() if n != exclude]
    if not items: return []
    d = [np.linalg.norm((f - m['f']) / SD) for _, m in items]
    out = []
    for j in np.argsort(d)[:k]:
        for p in items[j][1]['progs']:
            if tuple(p) not in out: out.append(tuple(p))
    return out

def run_config(tag, tasks, feats, policy, memory=None, budget=BUDGET, pool=None, lam=0.0, depth=3, induce=False):
    cached = load(tag)
    if cached: return cached
    jobs = []
    for n, t in tasks.items():
        lp = np.log(np.ones(P) / P) if policy is None else policy.logp(feats[n])
        extra = retrieve(feats[n], memory, exclude=n) if memory else ()
        jobs.append((n, t, lp, budget, extra, lam, depth, induce))
    t0 = time.time()
    res = dict(pool.imap_unordered(solve_one, jobs, chunksize=4))
    res['_meta'] = {'wall': time.time() - t0}
    save(tag, res); 
    s = sum(v['solved'] for k, v in res.items() if k != '_meta'); ver = sum(v['verified'] for k, v in res.items() if k != '_meta')
    print(f'{tag}: solved {s}/{len(tasks)} verified {ver} wall {res["_meta"]["wall"]:.0f}s', flush=True)
    return res

def mem_from(res, feats):
    return {n: {'f': feats[n], 'progs': r['progs']} for n, r in res.items() if n != '_meta' and r['verified']}
def data_from(mem): return [(m['f'], tuple(m['progs'][0])) for m in mem.values() if m['progs'][0]]

if __name__ == '__main__':
    pool = Pool(2)
    # ---- Round 0 on training tasks (Stage 1 only)
    r0 = run_config('train_r0_uniform', tr, FT, None, pool=pool)
    mem0 = mem_from(r0, FT); d0 = data_from(mem0)
    pi1 = Policy(MU, SD); pi1.fit(d0)                                  # Stage 2
    # ---- Evaluation ablations (held-out ARC evaluation set)
    A = run_config('eval_A_S1', ev, FE, None, pool=pool)
    B = run_config('eval_B_S1S2', ev, FE, pi1, pool=pool)
    C = run_config('eval_C_S1S2S3', ev, FE, pi1, mem0, pool=pool)
    # ---- Stage 4: expert iteration round on training tasks (policy + retrieval, leave-self-out)
    r1 = run_config('train_r1_policy', tr, FT, pi1, mem0, pool=pool)
    mem1 = mem_from(r1, FT)
    for n, m in mem0.items():
        if n not in mem1: mem1[n] = m
    d1 = data_from(mem1)
    pi2 = Policy(MU, SD); pi2.fit(d1)
    D = run_config('eval_D_S1S2S3S4', ev, FE, pi2, mem1, pool=pool)
    # also: S2 trained on r1 data without retrieval (isolates S4)
    D2 = run_config('eval_D2_S1S2S4_noRAG', ev, FE, pi2, None, pool=pool)
    # ---- Stage 5: sequential stream on evaluation tasks, 5 chunks
    names = list(ev); random.Random(0).shuffle(names)
    chunks = [names[i::5] for i in range(5)]
    base_nll = pi2.nll(d1)
    stream = {}
    for mode in ['naive', 'ewc']:
        if load(f'stream_{mode}'): stream[mode] = load(f'stream_{mode}'); continue
        pol = pi2.clone(); mem = dict(mem1); seen = list(d1); log = []
        pol.F = None; pol.anchor = None
        if mode == 'ewc': pol.consolidate(d1, lam=100.0)
        for ci, ch in enumerate(chunks):
            sub = {n: ev[n] for n in ch}
            res = run_config(f'stream_{mode}_chunk{ci}', sub, FE, pol, mem, pool=pool)
            solved = sum(res[n]['solved'] for n in ch)
            newm = mem_from({n: res[n] for n in ch}, FE)
            mem.update(newm)
            nd = data_from(newm)
            if nd:
                pol.fit(nd, steps=300, lr=0.3, ewc=(mode == 'ewc'))
                if mode == 'ewc': pol.consolidate(nd, lam=100.0)
            log.append({'chunk': ci, 'solved': solved, 'n': len(ch), 'verified_new': len(nd),
                        'nll_train_mem': pol.nll(d1), 'nll_new': pol.nll(nd) if nd else None, 'lam_eff': getattr(pol,'lam_eff',None)})
            print(mode, log[-1], flush=True)
        stream[mode] = {'log': log, 'base_nll': base_nll}; save(f'stream_{mode}', stream[mode])
    print('DONE', flush=True)
