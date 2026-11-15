import json, os, numpy as np
from multiprocessing import Pool
from arcpipe import *
import run_all as RA
CATS = {
 'geometry': ['rot90','rot180','rot270','flipud','fliplr','transpose','antitranspose'],
 'objects': ['crop_bbox','crop_largest','crop_smallest','keep_largest','keep_smallest','remove_largest','remove_noise','compact'],
 'scale_tile': ['up2','up3','down2','down3','tile22','tile12','tile21','tile33','self_kron','dedup','top','bottom','left','right'],
 'mirror_sym': ['mir_h','mir_h2','mir_v','mir_v2','mir4','sym_lr','sym_ud','sym_both','sym_rot'],
 'physics_fill': ['g_down','g_up','g_left','g_right','fill_enclosed','hollow','halo'],
 'colour': [f'rm{c}' for c in range(1,10)]+[f'keep{c}' for c in range(1,10)]+['most1x1','least1x1'],
}
assert sorted(sum(CATS.values(),[]))==sorted(NAMES), set(NAMES)^set(sum(CATS.values(),[]))
if __name__=='__main__':
    pool=Pool(2)
    for b in [250,500,2000]:
        RA.run_config(f'extra_budget_{b}', RA.ev, RA.FE, None, budget=b, pool=pool)
    for cat,ps in CATS.items():
        lp=np.full(P,-np.inf)
        keep=[i for i,n in enumerate(NAMES) if n not in ps]; lp[keep]=np.log(1/len(keep))
        class U:  # fixed policy object returning custom logp
            def logp(self,f): return lp
        RA.run_config(f'extra_loco_{cat}', RA.ev, RA.FE, U(), budget=500, pool=pool)
    print('DONE',flush=True)
