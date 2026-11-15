import os
os.environ['ARC_DSL'] = 'v2'; os.environ['ARC_RES'] = '/home/claude/results_v3'
import json, time
from multiprocessing import Pool
import run_all as RA
from arcpipe import *
if __name__ == '__main__':
    pool = Pool(2)
    for tag, depth, width in [('beam_w100_d3', 3, 100), ('beam_w100_d5', 5, 100)]:
        if os.path.exists(f'{RA.R}/eval_{tag}.json'): continue
        t0 = time.time()
        res = dict(pool.imap_unordered(solve_beam, [(n, t, width, depth, 4.0) for n, t in RA.ev.items()], chunksize=4))
        res['_meta'] = {'wall': time.time() - t0}; json.dump(res, open(f'{RA.R}/eval_{tag}.json', 'w'))
        print(tag, 'solved', sum(v['solved'] for k, v in res.items() if k != '_meta'), 'verified', sum(v['verified'] for k, v in res.items() if k != '_meta'), 'wall', round(res['_meta']['wall']), flush=True)
    print('DONE', flush=True)
