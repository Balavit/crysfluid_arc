import os
os.environ['ARC_DSL'] = 'v1'; os.environ['ARC_RES'] = '/home/claude/results'
import numpy as np
from multiprocessing import Pool
import run_all as RA
from arcpipe import *
if __name__ == '__main__':
    pool = Pool(2)
    for tag, split in [('arc2_eval', 'arc2_eval'), ('concept', 'concept')]:
        T = load_tasks(split); F = {k: features(v) for k, v in T.items()}
        RA.run_config(f'{tag}_v1', T, F, None, budget=500, pool=pool)
    print('DONE', flush=True)
