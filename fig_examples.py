import os, json
os.environ['ARC_DSL'] = 'v2'
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from arcpipe import load_tasks, search, score
import run_all as RA
CM = ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
ev = load_tasks('evaluation'); b = json.load(open('/home/claude/results_v3/eval_v3_ind_b250.json'))
bad = [k for k, v in b.items() if k != '_meta' and v['verified'] and not v['solved']]
print('verified-but-wrong:', bad, [b[k]['progs'][0] for k in bad])
rows = [('332efdb3', '1: S1b rule'), ('0c786b71', '2: S1 program'), ('5d2a5c43', '3: S1 program'), (bad[0], '4: failure')]
fig, axs = plt.subplots(len(rows), 4, figsize=(4.8, 0.98 * len(rows)), dpi=250)
for r, (k, lab) in enumerate(rows):
    t = ev[k]; cands, _ = search(t, np.log(np.ones(RA.P) / RA.P), 250, 3, [], lam=2.0, induce=True)
    di, do = t['train'][0]; ti, to = t['test'][0]
    pred = cands[0][2][0]; ok = np.array_equal(pred, to); prog = ' > '.join(cands[0][0]) + (' + colour map' if cands[0][1] else '')
    for c, (g, ttl) in enumerate([(di, 'demo input'), (do, 'demo output'), (ti, 'test input'), (pred, 'ours: ' + ('correct' if ok else 'wrong'))]):
        a = axs[r, c]; a.imshow(g, cmap=CM, vmin=0, vmax=9); a.set_xticks(np.arange(-.5, g.shape[1], 1), minor=True); a.set_yticks(np.arange(-.5, g.shape[0], 1), minor=True)
        a.grid(which='minor', color='#555', lw=0.25); a.tick_params(which='both', length=0, labelbottom=False, labelleft=False)
        for s in a.spines.values(): s.set_linewidth(0.5)
        a.set_title(ttl, fontsize=7.5, pad=2, color=('#1a7f37' if (c == 3 and ok) else '#b42318' if c == 3 else 'black'))
    axs[r, 0].set_ylabel(lab, fontsize=7, rotation=90, ha='center', va='center', labelpad=4)
plt.tight_layout(w_pad=0.3, h_pad=0.5); plt.savefig('/home/claude/fig_examples.png', bbox_inches='tight', pad_inches=0.04)
