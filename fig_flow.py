import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
def mk(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize, dpi=250); ax.set_xlim(-1, xlim + 1); ax.set_ylim(0, ylim); ax.axis('off'); return fig, ax
def box(ax, x, y, w, h, t, fc, fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=1.2', fc=fc, ec='#333', lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, linespacing=1.3, fontweight='bold' if bold else 'normal')
def arr(ax, a, b, c='#333', ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9, lw=1.0, color=c, ls=ls, shrinkA=0, shrinkB=0))
def line(ax, pts, c='#c0392b', ls='-'):
    for i in range(len(pts) - 1):
        if i == len(pts) - 2: arr(ax, pts[i], pts[i + 1], c=c, ls=ls)
        else: ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], color=c, lw=1.0, ls=ls)
# ---- per-task inference flow
fig, ax = mk((4.8, 3.9), 100, 78)
box(ax, 0, 29, 14, 20, 'ARC\ntask:\ndemos +\ntest', '#eeeeee', bold=True)
ax.text(36, 76, 'Branch A: enumerate programs (S1+S2+S3)', fontsize=7.5, ha='center', color='#1f4e79', fontweight='bold')
box(ax, 18, 46, 37, 26, 'S3 seeds the search with\n5 nearest solved tasks\nS2 policy + heuristic\norder the search\nS1 best-first search,\n116 primitives, depth <= 3', '#dbe9fb', fs=6.8)
box(ax, 59, 49, 20, 20, 'Verify (S4):\nreproduces\nALL demo\noutputs?', '#f4f0c8', fs=6.8)
ax.text(40, 38.5, 'Branch B: induce a rule (S1b)', fontsize=7.5, ha='center', color='#1f4e79', fontweight='bold')
box(ax, 18, 8, 37, 26, 'Fit a lookup table\nfeature(cell) -> colour\n13 feature sets,\nsimple to complex', '#dbe9fb', fs=6.8)
box(ax, 59, 11, 20, 20, 'Gate (S4):\nconsistent,\ncovers test,\nLOO pass?', '#f4f0c8', fs=6.8)
box(ax, 83, 22, 17, 34, 'Pool\nranked:\nverified\nenumerated,\nthen LOO-\npassing rules;\nkeep 2\n(pass@2)', '#ece8fb', fs=6.5)
arr(ax, (14.2, 43), (17.8, 54)); arr(ax, (14.2, 35), (17.8, 22))
arr(ax, (55.2, 59), (58.8, 59)); arr(ax, (55.2, 21), (58.8, 21))
arr(ax, (79.2, 59), (82.8, 50)); arr(ax, (79.2, 21), (82.8, 29))
ax.text(79.5, 62.5, 'yes', fontsize=6.5, ha='left', color='#555'); ax.text(79.5, 14.5, 'yes', fontsize=6.5, ha='left', color='#555')
ax.text(69, 44, 'no: next node', fontsize=6.5, ha='center', color='#555'); ax.text(69, 4, 'no: discard', fontsize=6.5, ha='center', color='#555')
plt.savefig('/home/claude/fig_flow.png', bbox_inches='tight', pad_inches=0.04); plt.close()
# ---- learning loops
fig, ax = mk((4.8, 3.3), 100, 66)
ax.text(0, 64, 'Offline: expert iteration (Stages 2-4)', fontsize=7.5, color='#1f4e79', fontweight='bold')
bw = 22
box(ax, 0, 43, bw, 17, 'Training\ntasks\n(demos only)', '#eeeeee', bold=True)
box(ax, 26, 43, bw, 17, 'Solve with\nS1 + S2 + S3\n(per-task flow)', '#dbe9fb')
box(ax, 52, 43, bw, 17, 'Keep only\nverified\nprograms\n(S4 gate)', '#f4f0c8')
box(ax, 78, 43, bw, 17, 'Refit S2 policy,\ngrow S3\nmemory', '#fbe3d6')
for a, b in [((22.3, 51.5), (25.7, 51.5)), ((48.3, 51.5), (51.7, 51.5)), ((74.3, 51.5), (77.7, 51.5))]: arr(ax, a, b)
line(ax, [(89, 42.6), (89, 37.5), (37, 37.5), (37, 42.6)])
ax.text(63, 39.2, 'next round: better S2, larger S3', fontsize=6.8, color='#c0392b', ha='center')
ax.text(0, 31, 'Online: consolidation on a task stream (Stage 5)', fontsize=7.5, color='#1f4e79', fontweight='bold')
bw2 = 30
box(ax, 0, 1, bw2, 25, 'Stream of\nnew tasks:\nsolve, keep\nverified programs', '#eeeeee', bold=True)
box(ax, 35, 1, bw2, 25, 'Update S2 policy:\nnaive | EWC\n(clamped) |\nrehearsal of S1', '#dff3e8')
box(ax, 70, 1, bw2, 25, 'Measure:\nretention (old\ntasks) vs\nplasticity (new)', '#ece8fb')
arr(ax, (30.3, 13.5), (34.7, 13.5)); arr(ax, (65.3, 13.5), (69.7, 13.5))
plt.savefig('/home/claude/fig_loop.png', bbox_inches='tight', pad_inches=0.04); plt.close()
