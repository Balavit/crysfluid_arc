"""DSL v3: object-level primitives added on top of v2 (ARC_DSL=v3). Background colour is 0."""
import numpy as np
from scipy import ndimage

S8 = np.ones((3, 3), int)

def _objs(g, conn8=True):
    lab, n = ndimage.label(g != 0, structure=S8 if conn8 else None)
    return lab, n

def obj_gravity(dy, dx):
    """move every 8-connected object rigidly until it hits the border or another object (process nearest-first)."""
    def f(g):
        lab, n = _objs(g)
        if n == 0 or n > 30: return None
        out = g.copy(); h, w = g.shape
        order = list(range(1, n + 1))
        cells = {i: np.argwhere(lab == i) for i in order}
        key = (lambda i: -(cells[i][:, 0].max() * dy + cells[i][:, 1].max() * dx)) if (dy + dx) > 0 else (lambda i: (cells[i][:, 0].min() * -dy + cells[i][:, 1].min() * -dx))
        order.sort(key=key)
        for i in order:
            c = cells[i]; vals = g[c[:, 0], c[:, 1]]
            out[c[:, 0], c[:, 1]] = 0
            y, x = c[:, 0].copy(), c[:, 1].copy()
            while True:
                ny, nx = y + dy, x + dx
                if ny.min() < 0 or nx.min() < 0 or ny.max() >= h or nx.max() >= w or (out[ny, nx] != 0).any(): break
                y, x = ny, nx
            out[y, x] = vals
        return out
    return f

def connect(mode):
    """draw lines between same-coloured pixels sharing a row ('h'), a column ('v') or either ('hv') over background."""
    def f(g):
        out = g.copy(); h, w = g.shape
        if mode in ('h', 'hv'):
            for r in range(h):
                for c in np.unique(g[r][g[r] != 0]):
                    xs = np.where(g[r] == c)[0]
                    if len(xs) >= 2:
                        seg = out[r, xs[0]:xs[-1] + 1]; seg[seg == 0] = c
        if mode in ('v', 'hv'):
            for k in range(w):
                for c in np.unique(g[:, k][g[:, k] != 0]):
                    ys = np.where(g[:, k] == c)[0]
                    if len(ys) >= 2:
                        seg = out[ys[0]:ys[-1] + 1, k]; seg[seg == 0] = c
        return out if not np.array_equal(out, g) else None
    return f

def diag_rays(g):
    out = g.copy(); h, w = g.shape
    for y, x in np.argwhere(g != 0):
        for dy, dx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            yy, xx = y + dy, x + dx
            while 0 <= yy < h and 0 <= xx < w:
                if out[yy, xx] == 0: out[yy, xx] = g[y, x]
                yy += dy; xx += dx
    return out

def _cells(g):
    """split by full separator rows/cols of one colour; return list of subgrids or None."""
    h, w = g.shape
    for c in np.unique(g):
        if c == 0: continue
        rs = [r for r in range(h) if (g[r] == c).all()]; cs = [k for k in range(w) if (g[:, k] == c).all()]
        if not rs and not cs: continue
        rb = [-1] + rs + [h]; cb = [-1] + cs + [w]; out = []
        for a, b in zip(rb[:-1], rb[1:]):
            for p, q in zip(cb[:-1], cb[1:]):
                if b - a > 1 and q - p > 1: out.append(g[a + 1:b, p + 1:q])
        if 2 <= len(out) <= 30: return out
    return None

def cell_pick(kind):
    def f(g):
        cs = _cells(g)
        if cs is None: return None
        nz = [int((c != 0).sum()) for c in cs]
        if kind == 'most': i = int(np.argmax(nz))
        elif kind == 'least': i = int(np.argmin(nz))
        else:
            ks = [c.tobytes() + bytes(c.shape) for c in cs]
            u = [i for i, k in enumerate(ks) if ks.count(k) == 1]
            if len(u) != 1: return None
            i = u[0]
        return cs[i]
    return f

def color_hist(g):
    v = g[g != 0]
    if len(v) == 0: return None
    cnt = np.bincount(v, minlength=10); cols = [c for c in np.argsort(-cnt, kind='stable') if cnt[c] > 0][:10]
    return np.array(cols).reshape(-1, 1)

def color_bars(g):
    v = g[g != 0]
    if len(v) == 0: return None
    cnt = np.bincount(v, minlength=10); cols = [c for c in np.argsort(-cnt, kind='stable') if cnt[c] > 0][:10]
    m = cnt[cols].max()
    if m > 30: return None
    out = np.zeros((len(cols), m), int)
    for i, c in enumerate(cols): out[i, :cnt[c]] = c
    return out

def swap_top2(g):
    v = g[g != 0]
    if len(v) == 0: return None
    cnt = np.bincount(v, minlength=10); cols = [c for c in np.argsort(-cnt, kind='stable') if cnt[c] > 0]
    if len(cols) != 2: return None
    a, b = cols; out = g.copy(); out[g == a] = b; out[g == b] = a; return out

def invert_fg(g):
    v = np.unique(g[g != 0])
    if len(v) != 1: return None
    return np.where(g == 0, v[0], 0)

def outline_objs(g):
    lab, n = _objs(g)
    if n == 0: return None
    er = ndimage.binary_erosion(g != 0, structure=S8, border_value=0)
    return np.where(er, 0, g)

def crop_multicolor(g):
    lab, n = _objs(g)
    if n < 2 or n > 30: return None
    best, bs = None, -1
    for i in range(1, n + 1):
        m = lab == i; k = len(np.unique(g[m]))
        if k > bs: bs, best = k, m
    ys, xs = np.where(best); return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def recolor_by_size_desc(g):
    lab, n = _objs(g)
    if n < 2 or n > 9: return None
    sizes = ndimage.sum(g != 0, lab, range(1, n + 1)); order = np.argsort(-sizes, kind='stable'); out = g.copy()
    for rank, i in enumerate(order): out[lab == i + 1] = rank + 1
    return out

def flood_bg_from_border(g):
    """colour every background cell NOT reachable from the border with the most frequent colour (like fill_enclosed but colour-aware)."""
    bgm = g == 0; lab, n = ndimage.label(bgm)
    edge = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1]); v = g[g != 0]
    if len(v) == 0: return None
    c = np.argmax(np.bincount(v)); out = g.copy()
    for i in range(1, n + 1):
        if i not in edge: out[lab == i] = c
    return out if not np.array_equal(out, g) else None

V3 = {'gravity_obj_d': obj_gravity(1, 0), 'gravity_obj_u': obj_gravity(-1, 0), 'gravity_obj_l': obj_gravity(0, -1), 'gravity_obj_r': obj_gravity(0, 1),
      'connect_h': connect('h'), 'connect_v': connect('v'), 'connect_hv': connect('hv'), 'diag_rays': diag_rays,
      'cell_most': cell_pick('most'), 'cell_least': cell_pick('least'), 'cell_unique': cell_pick('unique'),
      'color_hist': color_hist, 'color_bars': color_bars, 'swap_top2': swap_top2, 'invert_fg': invert_fg,
      'outline_objs': outline_objs, 'crop_multicolor': crop_multicolor, 'recolor_size_desc': recolor_by_size_desc, 'fill_enclosed_c': flood_bg_from_border}
