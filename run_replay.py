"""Stage-5 replay study: stream dynamics of the policy using the demonstration-verified programs found on the
evaluation tasks by config D (no test labels). 10 random stream orders x several update rules."""
import json, random, numpy as np
from arcpipe import *
import run_all as RA
R='/home/claude/results/'
r0=RA.load('train_r0_uniform'); r1=RA.load('train_r1_policy'); D=RA.load('eval_D_S1S2S3S4')
mem0=RA.mem_from(r0,RA.FT); mem1=RA.mem_from(r1,RA.FT)
for n,m in mem0.items(): mem1.setdefault(n,m)
d1=RA.data_from(mem1); pi2=Policy(RA.MU,RA.SD); pi2.fit(d1)
print('base nll',pi2.nll(d1),'n d1',len(d1))
memE=RA.mem_from(D,RA.FE)                       # verified eval tasks (label-free)
items={n:(m['f'],tuple(m['progs'][0])) for n,m in memE.items() if m['progs'][0]}
names=sorted(RA.ev); print('verified eval tasks',len(items))
LR=0.3
def run(mode, seed, lam0=100.0, clamp=0.5):
    rng=random.Random(seed); order=names[:]; rng.shuffle(order); chunks=[order[i::5] for i in range(5)]
    pol=pi2.clone(); pol.F=None; pol.anchor=None
    if mode.startswith('ewc'): pol.consolidate(d1,lam=lam0)
    seen=[]; ret=[]; plas=[]
    for ch in chunks:
        nd=[items[n] for n in ch if n in items]
        if nd:
            if mode=='frozen': pass
            elif mode=='replay': pol.fit(nd+d1,steps=300,lr=LR)
            elif mode=='naive': pol.fit(nd,steps=300,lr=LR)
            else:
                if clamp is None:   # no stability clamp: reproduce the unstable variant
                    pol.lam=lam0
                    with np.errstate(all='ignore'):
                        for _ in range(300):
                            gW,gb,_n=pol._grad(nd,1e-3)
                            gW=gW+pol.lam*pol.F[0]*(pol.W-pol.anchor[0]); gb=gb+pol.lam*pol.F[1]*(pol.b-pol.anchor[1])
                            pol.W-=LR*gW; pol.b-=LR*gb
                else:
                    pol.lam=lam0
                    lam=min(lam0,clamp/(LR*max(pol.F[0].max(),pol.F[1].max(),1e-12)))
                    # same rule as Policy.fit but with a configurable clamp factor
                    for _ in range(300):
                        gW,gb,_n=pol._grad(nd,1e-3)
                        gW=gW+lam*pol.F[0]*(pol.W-pol.anchor[0]); gb=gb+lam*pol.F[1]*(pol.b-pol.anchor[1])
                        pol.W-=LR*gW; pol.b-=LR*gb
                pol.consolidate(nd,lam=lam0)
            with np.errstate(all='ignore'):
                plas.append(float(pol.nll(nd)))
        with np.errstate(all='ignore'):
            ret.append(float(pol.nll(d1)))
    return ret[-1], float(np.nanmean(plas)) if plas else float('nan')
cfgs=[('frozen',{}),('naive',{}),('replay',{}),
      ('ewc lam=10 c=0.5',dict(mode='ewc',lam0=10.0,clamp=0.5)),('ewc lam=100 c=0.5',dict(mode='ewc',lam0=100.0,clamp=0.5)),
      ('ewc lam=1000 c=0.5',dict(mode='ewc',lam0=1000.0,clamp=0.5)),
      ('ewc lam=100 c=0.1',dict(mode='ewc',lam0=100.0,clamp=0.1)),('ewc lam=100 c=0.9',dict(mode='ewc',lam0=100.0,clamp=0.9)),
      ('ewc lam=100 no clamp',dict(mode='ewc',lam0=100.0,clamp=None))]
out={}
for name,kw in cfgs:
    mode=kw.pop('mode',name); res=[run(mode,s,**kw) for s in range(10)]
    ret=np.array([r[0] for r in res]); pl=np.array([r[1] for r in res])
    out[name]={'ret':ret.tolist(),'plas':pl.tolist()}
    fin=np.isfinite(ret)
    print(f'{name:24s} final NLL(S1 memory) {np.mean(ret[fin]):.3f} +- {np.std(ret[fin]):.3f} (finite {fin.sum()}/10) | NLL(new) {np.nanmean(pl):.3f}',flush=True)
json.dump(out,open(R+'replay_study.json','w'))
