"""CrysFluid-ARC: CPU-scale 5-stage pipeline on ARC-AGI-1 (real data, real measurements).

Stage 1 (crystallized)   : 65-primitive grid DSL + best-first program search (uniform prior)
Stage 2 (meta-learning)  : task-conditioned softmax policy trained across tasks (learning-to-search)
Stage 3 (retrieval/ICL)  : kNN retrieval of solved tasks' programs, tried before search
Stage 4 (RL-style)       : expert iteration: search -> verified (demo-consistent) solutions -> retrain policy
Stage 5 (continual)      : online policy updates on a task stream, with / without EWC; forgetting measured
"""
import json, glob, os, time, heapq, itertools
import numpy as np
from scipy import ndimage

MAXS = 30

# ---------------------------------------------------------------- DSL
def _objs(g):
    lab, n = ndimage.label(g != 0, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        m = lab == i
        ys, xs = np.where(m)
        out.append((int(m.sum()), ys.min(), ys.max() + 1, xs.min(), xs.max() + 1, m))
    return out

def crop_bbox(g):
    ys, xs = np.where(g != 0)
    if len(ys) == 0: return None
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def _pick(g, largest):
    o = _objs(g)
    if not o: return None
    sizes = [x[0] for x in o]
    t = max(sizes) if largest else min(sizes)
    if sizes.count(t) > 1: return None
    return o[sizes.index(t)]

def crop_obj(largest):
    def f(g):
        o = _pick(g, largest)
        return None if o is None else g[o[1]:o[2], o[3]:o[4]]
    return f

def keep_obj(largest):
    def f(g):
        o = _pick(g, largest)
        return None if o is None else np.where(o[5], g, 0)
    return f

def remove_obj(largest):
    def f(g):
        o = _pick(g, largest)
        return None if o is None else np.where(o[5], 0, g)
    return f

def remove_noise(g):
    out = g.copy()
    for o in _objs(g):
        if o[0] == 1: out[o[5]] = 0
    return out

def _sink(g):
    out = np.zeros_like(g)
    for j in range(g.shape[1]):
        v = g[:, j][g[:, j] != 0]
        if len(v): out[-len(v):, j] = v
    return out

def grav(d):
    return {
        'down': _sink,
        'up': lambda g: np.flipud(_sink(np.flipud(g))),
        'right': lambda g: _sink(g.T).T,
        'left': lambda g: np.fliplr(_sink(np.fliplr(g).T).T),
    }[d]

def fill_enclosed(g):
    lab, n = ndimage.label(g == 0)
    border = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
    enc = (g == 0) & ~np.isin(lab, list(border))
    if not enc.any(): return None
    out = g.copy(); out[enc] = 10
    return out

def self_kron(g):
    h, w = g.shape
    if h * h > MAXS or w * w > MAXS: return None
    return np.kron((g != 0).astype(g.dtype), g)

def up(k):
    def f(g):
        if g.shape[0] * k > MAXS or g.shape[1] * k > MAXS: return None
        return np.kron(g, np.ones((k, k), dtype=g.dtype))
    return f

def down(k):
    def f(g):
        if g.shape[0] % k or g.shape[1] % k: return None
        return g[::k, ::k]
    return f

def tile(a, b):
    def f(g):
        if g.shape[0] * a > MAXS or g.shape[1] * b > MAXS: return None
        return np.tile(g, (a, b))
    return f

def _lim(x): return x if x.shape[0] <= MAXS and x.shape[1] <= MAXS else None

def mir4(g):
    t = np.hstack([g, np.fliplr(g)]); return _lim(np.vstack([t, np.flipud(t)]))

def half(which):
    def f(g):
        h, w = g.shape
        if which == 'top': r = g[:h // 2] if h // 2 else None
        elif which == 'bottom': r = g[h - h // 2:] if h // 2 else None
        elif which == 'left': r = g[:, :w // 2] if w // 2 else None
        else: r = g[:, w - w // 2:] if w // 2 else None
        return r
    return f

def sym(kind):
    def f(g):
        if kind == 'lr': return np.where(g != 0, g, np.fliplr(g))
        if kind == 'ud': return np.where(g != 0, g, np.flipud(g))
        if kind == 'both':
            a = np.where(g != 0, g, np.fliplr(g)); return np.where(a != 0, a, np.flipud(a))
        if kind == 'rot':
            return np.where(g != 0, g, np.rot90(g, 2))
    return f

def color1x1(most):
    def f(g):
        v = g[g != 0]
        if len(v) == 0: return None
        c = np.bincount(v, minlength=11)[1:]
        t = c.max() if most else c[c > 0].min()
        if (c == t).sum() > 1: return None
        return np.array([[int(np.argmax(c == t)) + 1]], dtype=g.dtype)
    return f

def rm_c(c): return lambda g: np.where(g == c, 0, g) if (g == c).any() else None
def keep_c(c): return lambda g: np.where(g == c, g, 0) if (g == c).any() else None

def hollow(g):
    nz = g != 0
    er = ndimage.binary_erosion(nz, structure=np.ones((3, 3)), border_value=0)
    if not er.any(): return None
    return np.where(er, 0, g)

def halo(g):
    nz = g != 0
    d = ndimage.binary_dilation(nz, structure=np.ones((3, 3))) & ~nz
    if not d.any(): return None
    out = g.copy(); out[d] = 10; return out

def dedup(g):
    r = [0] + [i for i in range(1, g.shape[0]) if not np.array_equal(g[i], g[i - 1])]
    g = g[r]
    c = [0] + [j for j in range(1, g.shape[1]) if not np.array_equal(g[:, j], g[:, j - 1])]
    return g[:, c]

def compact(g):
    r = np.where((g != 0).any(1))[0]; c = np.where((g != 0).any(0))[0]
    if len(r) == 0: return None
    return g[r][:, c]

PRIMS = {
    'rot90': lambda g: np.rot90(g, 1), 'rot180': lambda g: np.rot90(g, 2), 'rot270': lambda g: np.rot90(g, 3),
    'flipud': np.flipud, 'fliplr': np.fliplr, 'transpose': lambda g: g.T,
    'antitranspose': lambda g: np.rot90(g, 2).T,
    'crop_bbox': crop_bbox, 'crop_largest': crop_obj(True), 'crop_smallest': crop_obj(False),
    'keep_largest': keep_obj(True), 'keep_smallest': keep_obj(False),
    'remove_largest': remove_obj(True), 'remove_noise': remove_noise,
    'up2': up(2), 'up3': up(3), 'down2': down(2), 'down3': down(3),
    'tile22': tile(2, 2), 'tile12': tile(1, 2), 'tile21': tile(2, 1), 'tile33': tile(3, 3),
    'mir_h': lambda g: _lim(np.hstack([g, np.fliplr(g)])), 'mir_h2': lambda g: _lim(np.hstack([np.fliplr(g), g])),
    'mir_v': lambda g: _lim(np.vstack([g, np.flipud(g)])), 'mir_v2': lambda g: _lim(np.vstack([np.flipud(g), g])),
    'mir4': mir4,
    'top': half('top'), 'bottom': half('bottom'), 'left': half('left'), 'right': half('right'),
    'g_down': grav('down'), 'g_up': grav('up'), 'g_left': grav('left'), 'g_right': grav('right'),
    'fill_enclosed': fill_enclosed, 'self_kron': self_kron,
    'sym_lr': sym('lr'), 'sym_ud': sym('ud'), 'sym_both': sym('both'), 'sym_rot': sym('rot'),
    'most1x1': color1x1(True), 'least1x1': color1x1(False),
    'hollow': hollow, 'halo': halo, 'dedup': dedup, 'compact': compact,
}
for _c in range(1, 10):
    PRIMS[f'rm{_c}'] = rm_c(_c); PRIMS[f'keep{_c}'] = keep_c(_c)
NAMES = list(PRIMS)
P = len(NAMES)
IDX = {n: i for i, n in enumerate(NAMES)}

def apply_prim(name, g):
    try:
        r = PRIMS[name](g)
    except Exception:
        return None
    if r is None or r.size == 0 or r.shape[0] > MAXS or r.shape[1] > MAXS: return None
    return np.ascontiguousarray(r)

def run_prog(prog, g):
    for p in prog:
        g = apply_prim(p, g)
        if g is None: return None
    return g

# ---------------------------------------------------------------- verification (demonstrations only)
def check(states, targets):
    for s, t in zip(states, targets):
        if s.shape != t.shape: return False, None
    if all(np.array_equal(s, t) for s, t in zip(states, targets)): return True, {}
    S = np.concatenate([s.ravel() for s in states]).astype(np.int64)
    T = np.concatenate([t.ravel() for t in targets]).astype(np.int64)
    m = {}
    for v in np.unique(S * 11 + T):
        s, t = divmod(int(v), 11)
        if s in m and m[s] != t: return False, None
        m[s] = t
    return True, m

def apply_cmap(g, m):
    if not m: return g
    out = g.copy()
    for s, t in m.items(): out[g == s] = t
    return out

# ---------------------------------------------------------------- task features
def load_tasks(split):
    d = {}
    for f in sorted(glob.glob(f'/home/claude/data/ARC-AGI/data/{split}/*.json')):
        t = json.load(open(f))
        d[os.path.basename(f)[:-5]] = {
            'train': [(np.array(p['input'], dtype=np.int8), np.array(p['output'], dtype=np.int8)) for p in t['train']],
            'test': [(np.array(p['input'], dtype=np.int8), np.array(p['output'], dtype=np.int8)) for p in t['test']]}
    return d

NF = 19
def features(task):
    rows = []
    for i, o in task['train']:
        hi, wi = i.shape; ho, wo = o.shape
        ci = set(np.unique(i)) - {0}; co = set(np.unique(o)) - {0}
        crop = crop_bbox(i)
        sub = any(o.shape == (x[2] - x[1], x[4] - x[3]) and np.array_equal(i[x[1]:x[2], x[3]:x[4]], o) for x in _objs(i)) if True else False
        rows.append([
            float(i.shape == o.shape), np.log(ho / hi), np.log(wo / wi), np.log(o.size / i.size),
            len(ci) / 9, len(co) / 9, (i != 0).mean(), (o != 0).mean(),
            len(co - ci) / 9, len(ci - co) / 9, float(sub),
            float(crop is not None and crop.shape == o.shape and np.array_equal(crop, o)),
            float(np.array_equal(o, np.fliplr(o))), float(np.array_equal(o, np.flipud(o))),
            float(ho % hi == 0 and wo % wi == 0 and ho > hi), float(hi % ho == 0 and wi % wo == 0 and hi > ho),
            float(o.size == 1), np.log1p(len(_objs(i))) / 3,
            float((i != o).mean()) if i.shape == o.shape else 0.0])
    return np.mean(rows, 0)

# ---------------------------------------------------------------- search
def search(task, logp, budget=600, depth=3, extra=(), tmax=25.0):
    """Best-first (uniform-cost under policy -log p) search over primitive sequences with state de-duplication.
    Only the demonstration pairs are used for verification. Returns up to 2 candidate programs (distinct test outputs)."""
    ins = [i for i, _ in task['train']]; tgts = [o for _, o in task['train']]
    tests = [i for i, _ in task['test']]
    cands, seen_out = [], set()

    def try_prog(prog, states):
        ok, m = check(states, tgts)
        if not ok: return False
        outs = []
        for t in tests:
            r = run_prog(prog, t)
            if r is None: return False
            outs.append(apply_cmap(r, m))
        key = b''.join(o.tobytes() + bytes(o.shape) for o in outs)
        if key in seen_out: return False
        seen_out.add(key); cands.append((prog, m, outs)); return True

    # Stage-3 retrieved programs are verified first
    for prog in extra:
        states = [run_prog(prog, g) for g in ins]
        if any(s is None for s in states): continue
        try_prog(tuple(prog), states)
        if len(cands) >= 2: return cands, 0
    t0 = time.time()
    cnt = itertools.count()
    start = tuple(ins)
    if try_prog((), list(start)) and len(cands) >= 2: return cands, 0
    heap = [(0.0, next(cnt), (), start)]
    seen = {b''.join(g.tobytes() + bytes(g.shape) for g in start)}
    nodes = 0
    cost = -logp
    while heap and nodes < budget and len(cands) < 2 and time.time() - t0 < tmax:
        c, _, prog, states = heapq.heappop(heap)
        nodes += 1
        if len(prog) >= depth: continue
        for pi, name in enumerate(NAMES):
            if not np.isfinite(cost[pi]): continue
            ns = []
            for g in states:
                r = apply_prim(name, g)
                if r is None: break
                ns.append(r)
            else:
                key = b''.join(g.tobytes() + bytes(g.shape) for g in ns)
                if key in seen: continue
                seen.add(key)
                np_ = prog + (name,)
                if try_prog(np_, ns) and len(cands) >= 2: return cands, nodes
                if len(np_) < depth:
                    heapq.heappush(heap, (c + cost[pi], next(cnt), np_, tuple(ns)))
    return cands, nodes

def score(task, cands):
    """pass@2 per ARC convention: every test input must be matched by one of the (<=2) candidates."""
    ok = True
    for k, (_, o) in enumerate(task['test']):
        if not any(np.array_equal(c[2][k], o) for c in cands[:2]): ok = False
    return ok

def solve_one(args):
    name, task, logp, budget, extra = args
    t0 = time.time()
    cands, nodes = search(task, logp, budget, 3, extra)
    return name, {'solved': bool(cands and score(task, cands)),
                  'verified': bool(cands),                       # consistent with demonstrations
                  'progs': [list(c[0]) for c in cands], 'cmaps': [{int(k): int(v) for k, v in c[1].items()} for c in cands],
                  'nodes': nodes, 'sec': time.time() - t0}

# ---------------------------------------------------------------- Stage 2/4/5: softmax policy network
class Policy:
    """p(prim | task features) = softmax(W f + b); trained by max-likelihood on verified programs.
    Supports EWC (diagonal empirical Fisher) for online updates."""
    def __init__(self, mu=None, sd=None):
        self.W = np.zeros((P, NF)); self.b = np.zeros(P)
        self.mu, self.sd = mu, sd
        self.F = None; self.anchor = None; self.lam = 0.0

    def _z(self, f): return (f - self.mu) / self.sd

    def probs(self, f):
        z = self._z(f); l = self.W @ z + self.b; l -= l.max(); e = np.exp(l); return e / e.sum()

    def logp(self, f, mix=0.7):
        return np.log(mix * self.probs(f) + (1 - mix) / P)

    def _grad(self, data, l2):
        gW = np.zeros_like(self.W); gb = np.zeros_like(self.b); nll = 0; n = 0
        for f, prog in data:
            z = self._z(f); p = self.probs(f)
            for name in prog:
                y = np.zeros(P); y[IDX[name]] = 1
                d = p - y
                gW += np.outer(d, z); gb += d; nll -= np.log(p[IDX[name]] + 1e-12); n += 1
        n = max(n, 1)
        return gW / n + l2 * self.W, gb / n + l2 * self.b, nll / n

    def fisher(self, data):
        FW = np.zeros_like(self.W); Fb = np.zeros_like(self.b); k = 0
        for f, prog in data:
            gW, gb, _ = self._grad([(f, prog)], 0.0)
            FW += gW ** 2; Fb += gb ** 2; k += 1
        return FW / max(k, 1), Fb / max(k, 1)

    def fit(self, data, steps=300, lr=0.3, l2=1e-3, ewc=False):
        lam = self.lam
        if ewc and self.F is not None:   # stability clamp: lr*lam*Fmax <= 0.5 (otherwise the penalty step overshoots)
            lam = min(self.lam, 0.5 / (lr * max(self.F[0].max(), self.F[1].max(), 1e-12)))
        self.lam_eff = lam
        for _ in range(steps):
            gW, gb, _ = self._grad(data, l2)
            if ewc and self.F is not None:
                gW = gW + lam * self.F[0] * (self.W - self.anchor[0])
                gb = gb + lam * self.F[1] * (self.b - self.anchor[1])
            self.W -= lr * gW; self.b -= lr * gb

    def nll(self, data): return self._grad(data, 0.0)[2]

    def consolidate(self, data, lam):
        """Stage 5: store Fisher + anchor after learning `data` (accumulated online)."""
        FW, Fb = self.fisher(data)
        if self.F is None: self.F = (FW, Fb)
        else: self.F = (self.F[0] + FW, self.F[1] + Fb)
        self.anchor = (self.W.copy(), self.b.copy()); self.lam = lam

    def clone(self):
        q = Policy(self.mu, self.sd); q.W = self.W.copy(); q.b = self.b.copy()
        q.F = None if self.F is None else (self.F[0].copy(), self.F[1].copy())
        q.anchor = None if self.anchor is None else (self.anchor[0].copy(), self.anchor[1].copy()); q.lam = self.lam
        return q
