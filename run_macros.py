"""Library learning: promote distinct verified multi-step TRAINING programs to macro-primitives; evaluate once on ARC-AGI-1 eval."""
import os, json, sys
os.environ['ARC_DSL'] = 'v2'; os.environ['ARC_RES'] = '/home/claude/results_v2'
phase = sys.argv[1]
if phase == 'make':
    import run_all as RA
    r1 = RA.load('train_r1'); r0 = RA.load('train_r0')
    progs = []
    for r in (r0, r1):
        for k, v in r.items():
            if k != '_meta' and v['verified']:
                for p in v['progs']:
                    if len(p) >= 2 and p not in progs: progs.append(p)
    json.dump(progs, open('/home/claude/macros.json', 'w')); print('macros', len(progs))
else:
    os.environ['ARC_MACROS'] = '/home/claude/macros.json'
    from multiprocessing import Pool
    import run_all as RA
    from arcpipe import *
    if __name__ == '__main__':
        pool = Pool(2); print('P with macros', P, flush=True)
        RA.run_config('eval_v2_macros', RA.ev, RA.FE, None, budget=500, pool=pool, lam=2.0, depth=3)
        print('DONE', flush=True)
