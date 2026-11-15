import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(4.8, 3.65), dpi=250); ax.set_xlim(0, 100); ax.set_ylim(0, 77); ax.axis('off')
def box(x, y, w, h, title, sub, fc, ts=8, ss=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.5', fc=fc, ec='#333', lw=0.9))
    ax.text(x + w / 2, y + h - 3.6, title, ha='center', va='center', fontsize=ts, fontweight='bold')
    ax.text(x + w / 2, y + h / 2 - 3.0, sub, ha='center', va='center', fontsize=ss, linespacing=1.25)
def arr(a, b, c='#333', ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9, lw=1.1, color=c, ls=ls, shrinkA=0, shrinkB=0))
W, H = 30, 18; X = [1, 35, 69]
ax.add_patch(FancyBboxPatch((1, 69), 98, 6.5, boxstyle='round,pad=0.3,rounding_size=1.5', fc='#eeeeee', ec='#333', lw=0.9))
ax.text(50, 72.3, 'ARC task: 2-5 demonstration pairs + test input', ha='center', va='center', fontsize=8, fontweight='bold')
yA, yB, yC = 47, 24, 1
box(X[0], yA, W, H, 'S2 Meta-learned', 'task-conditioned\nsoftmax policy\n(learns to search)', '#fbe3d6')
box(X[1], yA, W, H, 'S1 Crystallized', '116 primitives,\nbest-first search,\npartial-credit score', '#dbe9fb')
box(X[2], yA, W, H, 'S3 Retrieval', 'k-NN over solved\ntasks seeds the\nsearch', '#dff3e8')
box(X[0], yB, W, H, 'S1b Induction', 'feature(cell) ->\ncolour tables\n(13 sets, LOO)', '#dbe9fb')
box(X[1], yB, W, H, 'Verifier', 'exact match on all\ndemonstration pairs\n(no test labels)', '#f4f0c8')
box(X[2], yB, W, H, 'Output', 'two distinct test\npredictions\n(pass@2)', '#ece8fb')
box(X[1], yC, W, 18, 'S4 Expert iter.', 'verified programs\nretrain S2 and\ngrow S3 memory', '#fbe3d6')
box(X[2], yC, W, 18, 'S5 Consolidation', 'online update with\nEWC or rehearsal,\nanchored to S1', '#dff3e8')
arr((50, 69), (50, yA + H + 0.6))
arr((X[0] + W + 0.5, yA + 9), (X[1] - 0.5, yA + 9)); arr((X[2] - 0.5, yA + 9), (X[1] + W + 0.5, yA + 9))
arr((50, yA - 0.2), (50, yB + H + 0.6))
arr((X[0] + W + 0.5, yB + 9), (X[1] - 0.5, yB + 9)); arr((X[1] + W + 0.5, yB + 9), (X[2] - 0.5, yB + 9))
arr((50, yB - 0.2), (50, yC + 18 + 0.6)); arr((X[1] + W + 0.5, yC + 9), (X[2] - 0.5, yC + 9))
ax.text(17, 10, 'only programs that\nare verified on the\ndemonstrations are\nlearned from', fontsize=7, va='center', ha='center', color='#444', style='italic')
plt.savefig('fig_arch.png', bbox_inches='tight', pad_inches=0.04)
