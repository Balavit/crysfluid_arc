import os, json, math, numpy as np
os.environ['ARC_DSL'] = 'v2'
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from arcpipe import load_tasks, features
R1 = '/home/claude/results/'; R2 = '/home/claude/results_v2/'; R3 = '/home/claude/results_v3/'; R4 = '/home/claude/results_v4/'
def L(path):
    return json.load(open(path)) if os.path.exists(path) else None
def tasks(r): return {k: v for k, v in r.items() if k != '_meta'}
def solved(r): return {k for k, v in tasks(r).items() if v['solved']}
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - w) / d, 100 * (c + w) / d
def mcnemar(a, b, universe):
    """exact two-sided sign test on discordant pairs"""
    n10 = len(a - b); n01 = len(b - a); n = n10 + n01
    if n == 0: return n10, n01, 1.0
    k = min(n10, n01); p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return n10, n01, min(1.0, p)

N = {}
cfg1 = {'v1_S1': R1 + 'eval_A_S1.json', 'v1_S1S2S3S4': R1 + 'eval_D_S1S2S3S4.json', 'v1_b250': R1 + 'extra_budget_250.json', 'v1_b500': R1 + 'extra_budget_500.json'}
cfg2 = {k: R2 + f + '.json' for k, f in {
    'v2_nolam': 'eval_v2_S1_nolam', 'v2_H': 'eval_v2_H', 'v2_H_S2S3': 'eval_v2_H_S2S3', 'v2_full': 'eval_v2_full', 'v2_H_S2': 'eval_v2_H_S2', 'v2_H_S3': 'eval_v2_H_S3',
    'v2_b250': 'scale_b250', 'v2_b50': 'scale_b50', 'v2_b100': 'scale_b100', 'v2_H_b250': 'eval_v2_H_b250', 'v2_b1000': 'scale_b1000', 'base_d0': 'eval_v2_base_d0', 'base_d1': 'eval_v2_base_d1'}.items()}
cfg3 = {'v3_ind_b250': 'eval_v3_ind_b250', 'v3_ind_b250_nogate': 'eval_v3_ind_b250_nogate', 'v3_full': 'eval_v3_full_b500', 'v3_beam_d3': 'eval_beam_w100_d3', 'v3_beam_d5': 'eval_beam_w100_d5'}
cfg3 = {k: R3 + f + '.json' for k, f in cfg3.items()}
cfg3['v4_b250'] = R4 + 'eval_v4_b250.json'
RES = {}
for k, pth in {**cfg1, **cfg2, **cfg3}.items():
    r = L(pth)
    if r: RES[k] = r
for k, r in RES.items():
    t = tasks(r); s = sum(v['solved'] for v in t.values()); v = sum(v['verified'] for v in t.values()); n = len(t)
    lo, hi = wilson(s, n)
    N[k] = {'solved': s, 'verified': v, 'n': n, 'acc': 100 * s / n, 'ci': [lo, hi], 'wrong_verified': v - s,
            'prec': (100 * s / v if v else None), 'nodes_mean': float(np.mean([x['nodes'] for x in t.values()])),
            'sec_median': float(np.median([x['sec'] for x in t.values()])), 'sec_p95': float(np.percentile([x['sec'] for x in t.values()], 95)),
            'sec_mean': float(np.mean([x['sec'] for x in t.values()]))}
# paired tests vs v1 and between v2 variants
pairs = [('v4_b250', 'v3_ind_b250'), ('v4_b250', 'v2_H_b250'), ('v3_ind_b250', 'v2_H_b250'), ('v3_ind_b250_nogate', 'v2_H_b250'), ('v3_ind_b250', 'v3_ind_b250_nogate'), ('v3_full', 'v2_full'), ('v3_full', 'v1_S1'), ('v3_beam_d3', 'v2_H'), ('v3_beam_d5', 'v3_beam_d3'), ('v2_nolam', 'v1_S1'), ('v2_H', 'v2_nolam'), ('v2_H_S2S3', 'v2_H'), ('v2_full', 'v2_H'), ('v2_full', 'v1_S1'), ('v2_full', 'v2_H_S2S3')]
N['paired'] = {}
for a, b in pairs:
    if a in RES and b in RES:
        n10, n01, p = mcnemar(solved(RES[a]), solved(RES[b]), None); N['paired'][f'{a} vs {b}'] = {'only_first': n10, 'only_second': n01, 'p': p}
# training results
for tag in ['train_r0', 'train_r1']:
    r = L(R2 + tag + '.json')
    if r: t = tasks(r); N[tag] = {'solved': sum(v['solved'] for v in t.values()), 'verified': sum(v['verified'] for v in t.values())}
for tag, f in [('v1_train_r0', R1 + 'train_r0_uniform.json'), ('v1_train_r1', R1 + 'train_r1_policy.json'), ('tune_lam0_d3', R2 + 'tune_lam0_d3.json')]:
    r = L(f)
    if r: t = tasks(r); N[tag] = {'solved': sum(v['solved'] for v in t.values()), 'verified': sum(v['verified'] for v in t.values())}
N['decisions'] = L(R2 + 'decisions.json')
tv = {k: L(R3 + f + '.json') for k, f in [('noind', 'train_v3_noind_b250'), ('ind', 'train_v3_ind_b250'), ('nogate', 'train_v3_ind_b250_nogate')]}
tv4 = L(R4 + 'train_v4_b250.json')
N['train_v4'] = {'solved': len(solved(tv4)), 'verified': sum(v['verified'] for v in tasks(tv4).values())}
n10, n01, p = mcnemar(solved(tv4), solved(tv['ind']), None); N['train_v4']['paired'] = {'only_first': n10, 'only_second': n01, 'p': p}
from collections import Counter as _C
N['v4_new_prims'] = dict(_C(n for k, v in tasks(RES['v4_b250']).items() if v['solved'] and k not in solved(RES['v3_ind_b250']) for n in (v['progs'][0] if v['progs'] else [])))
N['train_v3'] = {k: {'solved': len(solved(r)), 'verified': sum(v['verified'] for v in tasks(r).values())} for k, r in tv.items() if r}
if tv['noind'] and tv['ind']:
    n10, n01, p = mcnemar(solved(tv['ind']), solved(tv['noind']), None); N['train_v3']['paired_ind'] = {'only_first': n10, 'only_second': n01, 'p': p}
if tv['noind'] and tv['nogate']:
    n10, n01, p = mcnemar(solved(tv['nogate']), solved(tv['noind']), None); N['train_v3']['paired_nogate'] = {'only_first': n10, 'only_second': n01, 'p': p}
if 'v3_full' in RES:
    from collections import Counter
    ind_solved = [(k, v['progs'][0][0]) for k, v in tasks(RES['v3_full']).items() if v['solved'] and v['progs'] and v['progs'][0][0].startswith('induce')]
    N['v3_induced_solved'] = ind_solved
    N['v3_induced_unchecked'] = [k for k, n in ind_solved if n.endswith('*')]
# other datasets
for tag in ['arc2_eval', 'concept']:
    for var in ['v2_full', 'v2_S1_nolam', 'v3_full']:
        r = L((R3 if var == 'v3_full' else R2) + f'{tag}_{var}.json')
        if r:
            t = tasks(r); s = sum(v['solved'] for v in t.values()); n = len(t); N[f'{tag}_{var}'] = {'solved': s, 'n': n, 'acc': 100 * s / n, 'ci': list(wilson(s, n)), 'verified': sum(v['verified'] for v in t.values())}
    r = L(R1 + f'{tag}_v1.json')
    if r:
        t = tasks(r); s = sum(v['solved'] for v in t.values()); n = len(t); N[f'{tag}_v1'] = {'solved': s, 'n': n, 'acc': 100 * s / n, 'ci': list(wilson(s, n))}
# LOCO
loco = {}
ref = RES.get('v2_b250')
for f in sorted(os.listdir(R2)) if os.path.exists(R2) else []:
    if f.startswith('loco_'):
        r = L(R2 + f); loco[f[5:-5]] = sum(v['solved'] for v in tasks(r).values())
N['loco'] = loco; N['loco_ref'] = N.get('v2_b250', {}).get('solved')
# failure taxonomy on the headline config
head = RES.get('v4_b250') or RES.get('v3_full') or RES.get('v2_b1000') or RES.get('v2_full')
ev = load_tasks('evaluation'); N['taxonomy'] = {}
if head:
    def cat(t):
        sh = all(i.shape == o.shape for i, o in t['train'])
        if sh: return 'same size'
        if all(o.size == 1 for _, o in t['train']): return '1x1 output'
        if all(o.size < i.size for i, o in t['train']): return 'smaller output'
        if all(o.size > i.size for i, o in t['train']): return 'larger output'
        return 'mixed size'
    tx = {}
    for n, t in ev.items():
        c = cat(t); tx.setdefault(c, [0, 0]); tx[c][1] += 1; tx[c][0] += int(head[n]['solved'])
    N['taxonomy'] = tx
# verified-but-wrong cases and cmap usage in headline
if head:
    wr = [(k, v['progs'][0], v['cmaps'][0]) for k, v in tasks(head).items() if v['verified'] and not v['solved']]
    N['wrong_cases'] = wr
    sv = [(k, v['progs'][0], v['cmaps'][0]) for k, v in tasks(head).items() if v['solved']]
    N['solved_depth'] = {str(d): sum(1 for _, p, _ in sv if len(p) == d) for d in range(0, 5)}
    N['solved_cmap'] = sum(1 for _, _, m in sv if m)
    from collections import Counter
    N['top_prims'] = Counter(x for _, p, _ in sv for x in p).most_common(10)
    N['union_v1_v2'] = {'v1_only': len(solved(RES['v1_S1']) - solved(head)) if 'v1_S1' in RES else None,
                        'v2_only': len(solved(head) - solved(RES['v1_S1'])) if 'v1_S1' in RES else None}

# ---- pass@1, per-test-input accuracy, verification precision by number of demonstrations
from arcpipe import run_prog, apply_cmap
def cand_flags(t, v):
    out = []
    for prog, cm in zip(v['progs'], v['cmaps']):
        cm = {int(k): int(x) for k, x in cm.items()}; flags = []
        for ti, to in t['test']:
            r = run_prog(tuple(prog), ti)
            flags.append(False if r is None else bool(np.array_equal(apply_cmap(r, cm), to)))
        out.append(flags)
    return out
def dataset_stats(T, res):
    verified = [(k, v) for k, v in tasks(res).items() if v['verified']]
    p1 = 0; inp_total = 0; inp_hit1 = 0; inp_hit2 = 0; by = {}
    for k, v in verified:
        fl = cand_flags(T[k], v); nt = len(T[k]['test'])
        if fl and all(fl[0]): p1 += 1
        for j in range(nt):
            inp_total += 1; inp_hit1 += int(bool(fl) and fl[0][j]); inp_hit2 += int(any(f[j] for f in fl))
        nd = len(T[k]['train']); b = '2' if nd == 2 else ('3' if nd == 3 else '4+')
        by.setdefault(b, [0, 0]); by[b][0] += 1; by[b][1] += int(v['solved'])
    n_all_inputs = sum(len(T[k]['test']) for k in tasks(res))
    return {'pass1_tasks': p1, 'inputs_all': n_all_inputs, 'inputs_hit_top1': inp_hit1, 'inputs_hit_top2': inp_hit2, 'by_demos': by}
if head: N['stats_arc1'] = dataset_stats(ev, head)
cc = L(R3 + 'concept_v3_full.json') or L(R2 + 'concept_v2_full.json')
if cc: N['concept_headline'] = 'v3' if os.path.exists(R3 + 'concept_v3_full.json') else 'v2'
if cc: N['stats_concept'] = dataset_stats(load_tasks('concept'), cc)

# replay study
rp = L(R2 + 'replay_study_v2.json') or L(R1 + 'replay_study.json')
if rp:
    N['replay'] = {k: {'ret_mean': float(np.nanmean(np.array(v['ret'], float))), 'ret_sd': float(np.nanstd(np.array(v['ret'], float))),
                       'finite': int(np.isfinite(np.array(v['ret'], float)).sum()), 'plas_mean': float(np.nanmean(np.array(v['plas'], float)))} for k, v in rp.items()}
rt = L(R2 + 'retention.json')
if rt: N['retention'] = rt
rm = L(R2 + 'eval_v2_macros.json')
if rm:
    t = tasks(rm); s_ = sum(v['solved'] for v in t.values()); N['v2_macros'] = {'solved': s_, 'n': len(t), 'acc': 100 * s_ / len(t), 'ci': list(wilson(s_, len(t))), 'verified': sum(v['verified'] for v in t.values())}
    if 'v2_H' in RES:
        n10, n01, p_ = mcnemar(solved(rm), solved(RES['v2_H']), None); N['paired']['v2_macros vs v2_H'] = {'only_first': n10, 'only_second': n01, 'p': p_}
    N['n_macros'] = len(json.load(open('/home/claude/macros.json'))) if os.path.exists('/home/claude/macros.json') else None
if rp:
    ret = lambda k: np.array(rp[k]['ret'], float)
    N['replay_paired'] = {'ewc_lt_naive': int((ret('ewc lam=100 c=0.5') < ret('naive')).sum()), 'replay_lt_ewc': int((ret('replay') < ret('ewc lam=100 c=0.5')).sum()),
                          'ewc_reduction_pct': float(100 * (1 - (np.mean(ret('ewc lam=100 c=0.5')) - N['replay']['frozen']['ret_mean']) / (np.mean(ret('naive')) - N['replay']['frozen']['ret_mean'])))}
json.dump(N, open('/home/claude/numbers.json', 'w'), indent=1, default=str)

# ------------------------------------------------------------------ figures
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
def clean(a):
    a.spines[['top', 'right']].set_visible(False); a.tick_params(labelsize=8)
# Fig: scaling + ablation
fig, ax = plt.subplots(1, 2, figsize=(4.8, 2.2), dpi=220)
xs = [50, 100, 250, 500, 1000]
v1 = [None, None, N.get('v1_b250', {}).get('solved'), N.get('v1_b500', {}).get('solved'), N.get('v1_S1', {}).get('solved')]
v2 = [N.get('v2_b50', {}).get('solved'), N.get('v2_b100', {}).get('solved'), N.get('v2_b250', {}).get('solved'), N.get('v2_full', {}).get('solved'), N.get('v2_b1000', {}).get('solved')]
v1 = [np.nan if v is None else v for v in v1]; v2 = [np.nan if v is None else v for v in v2]
ax[0].plot(xs, v1, 'o-', color=C2, lw=1.8, label='DSL v1 (65)')
ax[0].plot(xs, v2, 'o-', color=C1, lw=1.8, label='v2 full (97)')
for x, a, b in zip(xs, v1, v2):
    if a == a: ax[0].text(x, a - 3, str(int(a)), ha='center', fontsize=7.5, color=C2)
    if b == b: ax[0].text(x, b + 1.5, str(int(b)), ha='center', fontsize=7.5, color=C1)
ax[0].set_xscale('log'); ax[0].set_xticks(xs); ax[0].set_xticklabels(xs); ax[0].set_xlabel('Search budget (node expansions)', fontsize=8)
ax[0].set_ylabel('Tasks solved of 400', fontsize=8); ax[0].set_ylim(0, 38); ax[0].legend(frameon=False, fontsize=7.5, loc='lower right'); ax[0].set_title('Budget scaling', fontsize=8.5); clean(ax[0])
labs = ['v1\nS1', 'v2\nS1', '+heur.', '+S2\n+S3', 'full\n+S4']
keys = ['v1_S1', 'v2_nolam', 'v2_H', 'v2_H_S2S3', 'v2_full']
vals = [N.get(k, {}).get('solved', 0) for k in keys]; cis = [N.get(k, {}).get('ci', [0, 0]) for k in keys]
ax[1].bar(range(5), vals, color=[C2, C1, C1, C1, C1], width=0.6)
for i, (v, c) in enumerate(zip(vals, cis)): ax[1].text(i, v + 1, str(v), ha='center', fontsize=8)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(labs, fontsize=8); ax[1].set_ylabel('Tasks solved of 400', fontsize=8)
ax[1].set_title('Stage ablation', fontsize=8.5); clean(ax[1]); plt.tight_layout(); plt.savefig('/home/claude/fig_scaling.png'); plt.close()
# Fig: LOCO
if loco and N['loco_ref']:
    fig, ax = plt.subplots(figsize=(3.9, 2.0), dpi=220)
    items = sorted(loco.items(), key=lambda kv: kv[1]); names = [k.replace('_', ' ') for k, _ in items]; d = [N['loco_ref'] - v for _, v in items]
    ax.barh(names, d, color=C1, height=0.6)
    for i, v in enumerate(d): ax.text(v + 0.15, i, str(v), va='center', fontsize=8)
    ax.set_xlabel('Tasks lost when family removed', fontsize=8); ax.tick_params(labelsize=8); clean(ax); plt.tight_layout(); plt.savefig('/home/claude/fig_loco.png'); plt.close()
# Fig: forgetting
if rp:
    order = ['frozen', 'naive', 'ewc lam=100 c=0.5', 'ewc lam=100 c=0.9', 'replay']
    names = ['frozen', 'naive', 'EWC\n.5', 'EWC\n.9', 'replay']
    fig, ax = plt.subplots(1, 2, figsize=(4.8, 2.2), dpi=220)
    m = [N['replay'][k]['ret_mean'] for k in order]; s = [N['replay'][k]['ret_sd'] for k in order]
    ax[0].bar(range(len(order)), m, yerr=s, color=[C4, C2, C1, C1, C3], width=0.6, capsize=2, error_kw={'lw': 0.8})
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(names, fontsize=8); ax[0].set_ylabel('NLL on Stage-1 memory', fontsize=8); ax[0].set_title('Forgetting (10 orders)', fontsize=8)
    p = [N['replay'][k]['plas_mean'] for k in order]
    ax[1].bar(range(len(order)), p, color=[C4, C2, C1, C1, C3], width=0.6); ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels(names, fontsize=8)
    ax[1].set_ylabel('NLL on newly learned tasks', fontsize=8); ax[1].set_title('Plasticity', fontsize=8)
    for a in ax: clean(a)
    plt.tight_layout(); plt.savefig('/home/claude/fig_forgetting.png'); plt.close()
print(json.dumps(N, indent=1, default=str)[:6000])
