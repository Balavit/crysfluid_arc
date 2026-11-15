import time, numpy as np
from arcpipe import *
tr = load_tasks('training'); print(len(tr), P)
names = list(tr)[:12]
logp = np.log(np.ones(P)/P)
t0=time.time(); s=0; nodes=0
for n in names:
    _, r = solve_one((n, tr[n], logp, 600, ()))
    s += r['solved']; nodes += r['nodes']; print(n, r['solved'], r['verified'], r['nodes'], round(r['sec'],1), r['progs'][:1])
print('solved', s, 'total sec', round(time.time()-t0,1))
