import os
os.environ['ARC_DSL'] = 'v3'; os.environ['ARC_RES'] = '/home/claude/results_v4'; os.environ['ARC_INDUCE_GATE'] = '1'
os.makedirs('/home/claude/results_v4', exist_ok=True)
from multiprocessing import Pool
import run_all as RA
from arcpipe import *
if __name__ == '__main__':
    pool = Pool(2); rc = RA.run_config
    a = rc('train_v4_b250', RA.tr, RA.FT, None, pool=pool, budget=250, lam=2.0, depth=3, induce=True)
    print('train', sum(v['solved'] for k, v in a.items() if k != '_meta'), flush=True)
    b = rc('eval_v4_b250', RA.ev, RA.FE, None, pool=pool, budget=250, lam=2.0, depth=3, induce=True)
    print('eval', sum(v['solved'] for k, v in b.items() if k != '_meta'), sum(v['verified'] for k, v in b.items() if k != '_meta'), flush=True)
    print('DONE', flush=True)
