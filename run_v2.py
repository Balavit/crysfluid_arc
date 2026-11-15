import os
os.environ['ARC_DSL'] = 'v2'; os.environ['ARC_RES'] = '/home/claude/results_v2'
import json, numpy as np
from multiprocessing import Pool
import run_all as RA
from arcpipe import *

FAM = {
 'geometry': ['rot90','rot180','rot270','flipud','fliplr','transpose','antitranspose'],
 'objects': ['crop_bbox','crop_largest','crop_smallest','keep_largest','keep_smallest','remove_largest','remove_noise','compact'],
 'scale_tile': ['up2','up3','down2','down3','tile22','tile12','tile21','tile33','self_kron','dedup','top','bottom','left','right','up_ncol'],
 'mirror_sym': ['mir_h','mir_h2','mir_v','mir_v2','mir4','sym_lr','sym_ud','sym_both','sym_rot','sym_t'],
 'physics_fill': ['g_down','g_up','g_left','g_right','fill_enclosed','hollow','halo'],
 'colour': [f'rm{c}' for c in range(1,10)]+[f'keep{c}' for c in range(1,10)]+['most1x1','least1x1','fill_majority'],
 'logic_grid': [n for n in NAMES if n.startswith('logic_') or n.startswith('overlay_')],
 'lines_rays': ['fill_rows','fill_cols','fill_cross','ray_right','ray_left','ray_down','ray_up','fill_bbox','draw_border','mark_largest','mark_smallest','roll_r','roll_l','roll_d','roll_u'],
}
assert sorted(sum(FAM.values(), [])) == sorted(NAMES), set(NAMES) ^ set(sum(FAM.values(), []))
cnt = lambda r: sum(v['solved'] for k, v in r.items() if k != '_meta')
rc = RA.run_config
if __name__ == '__main__':
    pool = Pool(2); dec = {}
    # ---- tuning on a half subset of TRAINING tasks only (every 2nd task); lam=0 reference taken from the cached full-training run
    sub = {n: t for i, (n, t) in enumerate(RA.tr.items()) if i % 2 == 0}
    base0 = RA.load('tune_lam0_d3'); tune = {'lam0_d3': sum(base0[n]['solved'] for n in sub)}
    for lam in [2, 4, 8]:
        tune[f'lam{lam}_d3'] = cnt(rc(f'tuneS_lam{lam}_d3', sub, RA.FT, None, budget=500, pool=pool, lam=lam, depth=3))
    bl = max([0, 2, 4, 8], key=lambda l: (tune[f'lam{l}_d3'], -l))
    if bl == 0: bl = 2
    tune[f'lam{bl}_d4'] = cnt(rc(f'tuneS_lam{bl}_d4', sub, RA.FT, None, budget=500, pool=pool, lam=bl, depth=4))
    bd = 4 if tune[f'lam{bl}_d4'] > tune[f'lam{bl}_d3'] else 3
    dec = {'tune_train_subset200_solved_b500': tune, 'lam': bl, 'depth': bd}; json.dump(dec, open(RA.R + '/decisions.json', 'w')); print('DECISION', dec, flush=True)
    kw = dict(lam=bl, depth=bd); BM = 500
    rc('eval_v2_S1_nolam', RA.ev, RA.FE, None, pool=pool, budget=BM, lam=0, depth=3)
    rc('eval_v2_H', RA.ev, RA.FE, None, pool=pool, budget=BM, **kw)
    r0 = rc('train_r0', RA.tr, RA.FT, None, pool=pool, budget=BM, **kw)
    mem0 = RA.mem_from(r0, RA.FT); d0 = RA.data_from(mem0); pi1 = Policy(RA.MU, RA.SD); pi1.fit(d0)
    rc('eval_v2_H_S2S3', RA.ev, RA.FE, pi1, mem0, pool=pool, budget=BM, **kw)
    r1 = rc('train_r1', RA.tr, RA.FT, pi1, mem0, pool=pool, budget=BM, **kw)
    mem1 = RA.mem_from(r1, RA.FT)
    for n, m in mem0.items(): mem1.setdefault(n, m)
    d1 = RA.data_from(mem1); pi2 = Policy(RA.MU, RA.SD); pi2.fit(d1)
    full = rc('eval_v2_full', RA.ev, RA.FE, pi2, mem1, pool=pool, budget=BM, **kw)
    rc('scale_b1000', RA.ev, RA.FE, pi2, mem1, pool=pool, budget=1000, **kw)
    # ---- other ARC datasets (policy/memory fitted on ARC-AGI-1 training only)
    for tag, split in [('arc2_eval', 'arc2_eval'), ('concept', 'concept')]:
        T = load_tasks(split); F = {k: features(v) for k, v in T.items()}
        rc(f'{tag}_v2_full', T, F, pi2, mem1, pool=pool, budget=BM, **kw)
        rc(f'{tag}_v2_S1_nolam', T, F, None, pool=pool, budget=BM, lam=0, depth=3)
    # ---- trivial baselines (identity+colour map; depth 1)
    rc('eval_v2_base_d0', RA.ev, RA.FE, None, pool=pool, lam=0, depth=0)
    rc('eval_v2_base_d1', RA.ev, RA.FE, None, pool=pool, lam=0, depth=1)
    rc('scale_b250', RA.ev, RA.FE, pi2, mem1, pool=pool, budget=250, **kw)
    class U:
        def __init__(s, mask): s.mask = mask
        def logp(s, f):
            lp = pi2.logp(f).copy(); lp[s.mask] = -np.inf; return lp
    for fam, ps in FAM.items():
        mask = np.array([n in ps for n in NAMES])
        rc(f'loco_{fam}', RA.ev, RA.FE, U(mask), mem1, pool=pool, budget=250, **kw)
    for b in [50, 100]:
        rc(f'scale_b{b}', RA.ev, RA.FE, pi2, mem1, pool=pool, budget=b, **kw)
    rc('eval_v2_H_b250', RA.ev, RA.FE, None, pool=pool, budget=250, **kw)
    print('DONE', flush=True)
