import os, json
os.environ['ARC_DSL'] = 'v2'; os.environ['ARC_RES'] = '/home/claude/results_v3'; os.environ['ARC_INDUCE_GATE'] = '1'
from multiprocessing import Pool
import run_all as RA
from arcpipe import *
if __name__ == '__main__':
    pool = Pool(2); rc = RA.run_config; R2 = '/home/claude/results_v2/'
    r0 = json.load(open(R2 + 'train_r0.json')); r1 = json.load(open(R2 + 'train_r1.json'))
    mem0 = RA.mem_from(r0, RA.FT); mem1 = RA.mem_from(r1, RA.FT)
    for n, m in mem0.items(): mem1.setdefault(n, m)
    pi2 = Policy(RA.MU, RA.SD); pi2.fit(RA.data_from(mem1))
    for tag, split in [('arc2_eval', 'arc2_eval'), ('concept', 'concept')]:
        T = load_tasks(split); F = {k: features(v) for k, v in T.items()}
        r = rc(f'{tag}_v3_full', T, F, pi2, mem1, pool=pool, budget=500, lam=2.0, depth=3, induce=True)
        print(tag, sum(v['solved'] for k, v in r.items() if k != '_meta'), 'verified', sum(v['verified'] for k, v in r.items() if k != '_meta'), flush=True)
    print('DONE', flush=True)
