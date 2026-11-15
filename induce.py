"""Demonstration-driven rule induction (Stage 1b). For same-shape tasks, learn a lookup rule  feature(cell) -> output colour
from the demonstration pairs. Feature sets are tried from simple to complex; a rule is accepted only if it is consistent on all
demonstrations, changes something, and covers the test input; leave-one-demo-out (LOO) agreement ranks candidates (Stage-4 gate)."""
import os
import numpy as np
from scipy import ndimage
GATE = os.environ.get("ARC_INDUCE_GATE", "1") == "1"
B = 12
def _pad(g): return np.pad(g.astype(np.int64), 1, constant_values=11)
def _bits(stack): return sum((1 << np.clip(s, 0, 11)) for s in stack)
def _obj_feats(g):
    lab, n = ndimage.label(g != 0, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel(), minlength=n + 1); sizes[0] = 0
    sz = sizes[lab]
    mx = sizes[1:].max() if n else 0; mn = sizes[1:].min() if n else 0
    big = ((sz == mx) & (lab > 0)).astype(np.int64); small = ((sz == mn) & (lab > 0)).astype(np.int64)
    rect = np.zeros(g.shape, np.int64)
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i); rect[lab == i] = int(len(ys) == (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    return sz, big, small, rect
def keys(g, fs):
    P = _pad(g); c = P[1:-1, 1:-1]
    up, dn, lf, rt = P[:-2, 1:-1], P[2:, 1:-1], P[1:-1, :-2], P[1:-1, 2:]
    ul, ur, dl, dr = P[:-2, :-2], P[:-2, 2:], P[2:, :-2], P[2:, 2:]
    if fs == 'c': return c
    if fs == 'c_cnt8': return c * 10 + sum((x > 0) & (x < 11) for x in (up, dn, lf, rt, ul, ur, dl, dr))
    if fs == 'c_n4set': return c * 5000 + _bits((up, dn, lf, rt))
    if fs == 'c_n8set': return c * 5000 + _bits((up, dn, lf, rt, ul, ur, dl, dr))
    if fs in ('c_size', 'c_rank', 'c_rect'):
        sz, big, small, rect = _obj_feats(g)
        if fs == 'c_size': return c * 1000 + np.minimum(sz, 60)
        if fs == 'c_rank': return c * 10 + big * 2 + small
        return c * 10 + rect
    if fs == 'c_par2':
        h, w = g.shape; r = np.arange(h)[:, None] % 2; q = np.arange(w)[None, :] % 2
        return c * 10 + r * 2 + q
    if fs in ('c_row', 'c_col', 'c_rowcol'):
        nz = g != 0
        rm = _bits([np.where(nz, g, -1)[:, j:j + 1].repeat(g.shape[1], 1) for j in range(g.shape[1])]) if False else None
        rowm = np.zeros(g.shape, np.int64); colm = np.zeros(g.shape, np.int64)
        for i in range(g.shape[0]):
            rowm[i, :] = sum(1 << int(v) for v in set(g[i][g[i] != 0].tolist()))
        for j in range(g.shape[1]):
            colm[:, j] = sum(1 << int(v) for v in set(g[:, j][g[:, j] != 0].tolist()))
        if fs == 'c_row': return c * 5000 + rowm
        if fs == 'c_col': return c * 5000 + colm
        return (c * 5000 + rowm) * 5000 + colm
    if fs == 'n4': return (((c * B + up) * B + dn) * B + lf) * B + rt
    if fs == 'n8': return (((((((c * B + up) * B + dn) * B + lf) * B + rt) * B + ul) * B + ur) * B + dl) * B + dr
    raise KeyError(fs)
FEATS = ['c', 'c_cnt8', 'c_n4set', 'c_size', 'c_rank', 'c_rect', 'c_par2', 'c_row', 'c_col', 'c_n8set', 'c_rowcol', 'n4', 'n8']
SIMPLE = {'c', 'c_cnt8', 'c_n4set', 'c_size', 'c_rank', 'c_rect', 'c_par2'}

def _fit(ks, outs):
    k = np.concatenate([x.ravel() for x in ks]); o = np.concatenate([x.ravel() for x in outs]).astype(np.int64)
    pair = k * 11 + o; up_ = np.unique(pair)
    if len(np.unique(k)) != len(up_): return None
    return up_ // 11, up_ % 11            # sorted keys, outputs
def _apply(model, k, g):
    uk, uo = model; idx = np.searchsorted(uk, k.ravel()); idx = np.clip(idx, 0, len(uk) - 1)
    seen = uk[idx] == k.ravel(); out = np.where(seen, uo[idx], g.ravel()).reshape(g.shape)
    return out.astype(g.dtype), float(seen.mean())

def induce(ins, tgts, tests, max_out=2):
    """Returns candidate list [(name, [outs per test], loo_pass)] best first."""
    if not all(i.shape == t.shape for i, t in zip(ins, tgts)): return []
    res = []
    for fs in FEATS:
        try:
            ks = [keys(i, fs) for i in ins]
            m = _fit(ks, tgts)
            if m is None: continue
            if all(np.array_equal(i, t) for i, t in zip(ins, tgts)): continue
            outs, ok = [], True
            for t in tests:
                o, cov = _apply(m, keys(t, fs), t)
                if cov < 0.99: ok = False; break
                outs.append(o)
            if not ok: continue
            loo = False
            if len(ins) >= 2:
                loo = True
                for j in range(len(ins)):
                    mj = _fit([k for q, k in enumerate(ks) if q != j], [t for q, t in enumerate(tgts) if q != j])
                    if mj is None: loo = False; break
                    oj, cov = _apply(mj, ks[j], ins[j])
                    if cov < 1.0 or not np.array_equal(oj, tgts[j]): loo = False; break
            if GATE and fs not in SIMPLE and not loo: continue
            res.append((f'induce:{fs}', outs, loo))
        except Exception:
            continue
    res.sort(key=(lambda r: (not r[2], FEATS.index(r[0][7:]))) if GATE else (lambda r: FEATS.index(r[0][7:])))
    seen, out = set(), []
    for name, outs, loo in res:
        key = b''.join(o.tobytes() + bytes(o.shape) for o in outs)
        if key in seen: continue
        seen.add(key); out.append((name, outs, loo))
        if len(out) >= max_out: break
    return out
