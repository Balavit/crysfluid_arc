import json, math, os, sys
from docxlib import *
import paper_static as S
N = json.load(open('/home/claude/numbers.json'))
def g(k, f, d='n/a'):
    v = N.get(k)
    return d if v is None or f not in v or v[f] is None else v[f]
def cnt(k):
    v = N.get(k); return None if v is None else v['solved']
def pct(k):
    v = N.get(k); return 'n/a' if v is None else f"{v['acc']:.2f}"
def ci(k):
    v = N.get(k); return 'n/a' if v is None else f"{v['ci'][0]:.1f}–{v['ci'][1]:.1f}"
def prec(k):
    v = N.get(k); return 'n/a' if v is None or v['prec'] is None else f"{v['prec']:.0f}%"
def pv(a, b):
    r = N['paired'].get(f'{a} vs {b}');
    if not r: return 'n/a'
    p = r['p']; return f"{r['only_first']}/{r['only_second']}, p={p:.2f}" if p >= 0.01 else f"{r['only_first']}/{r['only_second']}, p<0.01"
H = 'v4_b250'

def results():
    x = h1('Results')
    x += h2('Stage ablation on ARC-AGI-1')
    rows = [['Configuration', 'Budget', 'Solved', 'Acc. % (95% CI)', 'Verified', 'Prec.', 'vs. previous*'],
            ['Identity + colour map (depth 0)', '–', str(cnt('base_d0')), f"{pct('base_d0')} ({ci('base_d0')})", str(g('base_d0', 'verified')), prec('base_d0'), '–'],
            ['All single primitives (depth 1)', '–', str(cnt('base_d1')), f"{pct('base_d1')} ({ci('base_d1')})", str(g('base_d1', 'verified')), prec('base_d1'), '–'],
            ['v1 base search (65 primitives)', '1000', str(cnt('v1_S1')), f"{pct('v1_S1')} ({ci('v1_S1')})", str(g('v1_S1', 'verified')), prec('v1_S1'), '–'],
            ['v2 base search (97 primitives)', '500', str(cnt('v2_nolam')), f"{pct('v2_nolam')} ({ci('v2_nolam')})", str(g('v2_nolam', 'verified')), prec('v2_nolam'), pv('v2_nolam', 'v1_S1')],
            ['+ partial-credit heuristic', '500', str(cnt('v2_H')), f"{pct('v2_H')} ({ci('v2_H')})", str(g('v2_H', 'verified')), prec('v2_H'), pv('v2_H', 'v2_nolam')],
            ['+ S2 policy + S3 retrieval', '500', str(cnt('v2_H_S2S3')), f"{pct('v2_H_S2S3')} ({ci('v2_H_S2S3')})", str(g('v2_H_S2S3', 'verified')), prec('v2_H_S2S3'), pv('v2_H_S2S3', 'v2_H')],
            ['+ S4 expert iteration (full)', '500', str(cnt('v2_full')), f"{pct('v2_full')} ({ci('v2_full')})", str(g('v2_full', 'verified')), prec('v2_full'), pv('v2_full', 'v2_H_S2S3')],
            ['v2 heuristic only', '250', str(cnt('v2_H_b250')), f"{pct('v2_H_b250')} ({ci('v2_H_b250')})", str(g('v2_H_b250', 'verified')), prec('v2_H_b250'), '–'],
            ['+ S1b rule induction (LOO gate)', '250', str(cnt('v3_ind_b250')), f"{pct('v3_ind_b250')} ({ci('v3_ind_b250')})", str(g('v3_ind_b250', 'verified')), prec('v3_ind_b250'), pv('v3_ind_b250', 'v2_H_b250')],
            ['+ DSL v3 (116 primitives), S1b', '250', str(cnt('v4_b250')), f"{pct('v4_b250')} ({ci('v4_b250')})", str(g('v4_b250', 'verified')), prec('v4_b250'), pv('v4_b250', 'v3_ind_b250')]]
    x += table('**Table 2.** Ablation on the 400 ARC-AGI-1 evaluation tasks (pass@2). Verified = programs consistent with all demonstrations; Prec. = solved / verified. *Discordant task counts (first only / second only) and exact sign-test p-value against the row above (the S1b row is compared with the v2 heuristic-only row at the same budget and the DSL v3 row with the S1b row; it uses search with the heuristic only, because the learned stages did not change the solved set and were not refitted for the larger library).', rows,
               [2100, 650, 560, 1190, 640, 520, 1258], sz=14)
    return x

def sectref(): return None

def fig_scaling_text():
    x = figure('/home/claude/fig_scaling.png', '**Fig. 3.** Left: tasks solved versus search budget (node expansions) for DSL v1 and the full v2 system. Right: stage ablation; v1 uses budget 1000, v2 budget 500.', 0.82)
    c = {k: cnt(k) for k in ['v2_b50', 'v2_b100', 'v2_b250', 'v2_full', 'v2_b1000']}
    x += p1a(f"The v2 system solves {c['v2_b50']} tasks at budget 50, {c['v2_b100']} at 100, {c['v2_b250']} at 250, {c['v2_full']} at 500 and {c['v2_b1000']} at 1000, against 15, 17 and 17 for v1 at 250, 500 and 1000 (Fig. 3, left). "
             'The curve is flat above a few hundred expansions: more search time does not help, because everything the library can express is found early and the rest lies outside the search space. '
             'This is the quantitative form of the paper\'s central finding that the library, not the search, is the bottleneck.')
    return x

def loco_text():
    L_ = N.get('loco') or {}; ref = N.get('loco_ref')
    x = figure('/home/claude/fig_loco.png', '**Fig. 4.** Tasks lost when one primitive family is removed from the full system (budget 250; reference ' + str(ref) + ' solved).', 0.75)
    if L_ and ref is not None:
        dr = sorted(((ref - v, k) for k, v in L_.items()), reverse=True)
        top = ', '.join(f"{k.replace('_', ' ')} ({d})" for d, k in dr[:3]); zero = [k.replace('_', ' ') for d, k in dr if d <= 0]
        x += p1a(f'Removing one family at a time shows where the solved tasks come from (Fig. 4). The largest losses are {top}. '
                 + (f'Removing {", ".join(zero)} costs nothing on this set, i.e. those primitives are not needed for any currently solved task. ' if zero else '')
                 + 'Because each evaluation task is solved by a short program built from few families, the solved set is a small, identifiable slice of ARC, and the unsolved remainder lies outside the library.')
    return x

def datasets_text():
    rows = [['Dataset (tasks)', 'v1 base', 'v2 base', 'v2 full', 'v2 full + S1b']]
    for tag, nm in [('arc1', 'ARC-AGI-1 eval (400)'), ('arc2_eval', 'ARC-AGI-2 eval (120)'), ('concept', 'ConceptARC (160)')]:
        if tag == 'arc1':
            rows.append([nm, f"{cnt('v1_S1')} ({pct('v1_S1')}%)", f"{cnt('v2_nolam')} ({pct('v2_nolam')}%)", f"{cnt('v2_full')} ({pct('v2_full')}%)", f"{cnt('v3_full')} ({pct('v3_full')}%)"])
        else:
            f = lambda k: 'n/a' if k not in N else f"{N[k]['solved']} ({N[k]['acc']:.1f}%)"
            rows.append([nm, f(f'{tag}_v1'), f(f'{tag}_v2_S1_nolam'), f(f'{tag}_v2_full'), f(f'{tag}_v3_full')])
    x = table('**Table 4.** Transfer to other ARC datasets at budget 500 (v1: DSL v1; learned components fitted on ARC-AGI-1 training only; S1b: with rule induction).', rows, [2300, 1200, 1200, 1200, 1500], sz=16)
    return x

def taxonomy_text():
    tx = N.get('taxonomy') or {}
    rows = [['Output relation to input', 'Tasks', 'Solved', 'Rate']]
    for k, (s_, n_) in sorted(tx.items(), key=lambda kv: -kv[1][1]):
        rows.append([k, str(n_), str(s_), f'{100 * s_ / n_:.1f}%'])
    return table('**Table 5.** Where the solved evaluation tasks are (DSL v3 with S1b, budget 250), by relation between demonstration inputs and outputs.', rows, [2800, 1000, 1000, 1000], sz=16)

def library_text():
    if 'v2_macros' not in N: return ''
    m = N['v2_macros']; d = m['solved'] - N['v2_H']['solved']
    x = h2('Library learning and deeper search')
    x += p1a(f"Beyond learning *where to search*, a system can learn *what to search over*. We promote the {N.get('n_macros')} distinct verified two- and three-step programs found on the training tasks to macro-primitives and rerun the evaluation set once (budget 500, otherwise identical to the heuristic variant). "
             f"The macro library solves {m['solved']} tasks ({m['acc']:.2f}%), a change of {d:+d} against {N['v2_H']['solved']} without macros ({pv('v2_macros', 'v2_H')}). "
             + ('Macros therefore add reach beyond the primitives at no cost in accuracy.' if d > 0 else 'Macros therefore do not add reach on this set: programs that solve training tasks rarely recur as sub-programs of evaluation tasks.'))
    return x

def forgetting_text():
    R_ = N.get('replay'); 
    if not R_: return ''
    rp = N['replay_paired']
    x = h2('Forgetting during a task stream')
    x += p1a('We replay the stream with the demonstration-verified programs found on the evaluation tasks by the full system, in ten random orders of five chunks each, and track two quantities: the policy\'s loss on its own Stage-1 training data (forgetting) and its loss on the newly learned tasks (plasticity). '
             f"Frozen, the policy has loss {R_['frozen']['ret_mean']:.2f} on Stage-1 memory. Naive online updating raises it to {R_['naive']['ret_mean']:.2f} ± {R_['naive']['ret_sd']:.2f}. "
             f"Stability-clamped EWC ({chr(955)} = 100, *c* = 0.5) raises it to {R_['ewc lam=100 c=0.5']['ret_mean']:.2f} ± {R_['ewc lam=100 c=0.5']['ret_sd']:.2f}, i.e. {rp['ewc_reduction_pct']:.0f}% less drift, and beats naive updating in {rp['ewc_lt_naive']} of 10 orders. "
             f"Rehearsal reaches {R_['replay']['ret_mean']:.2f} ± {R_['replay']['ret_sd']:.2f}, below the frozen policy because it also fits the new tasks, and beats EWC in {rp['replay_lt_ewc']} of 10 orders (Fig. 6, left).")
    x += figure('/home/claude/fig_forgetting.png', '**Fig. 6.** Ten stream orders. Left: loss on Stage-1 memory after the stream (error bars: one standard deviation). Right: loss on the newly learned tasks.', 0.82)
    sw = ['ewc lam=10 c=0.5', 'ewc lam=100 c=0.5', 'ewc lam=1000 c=0.5', 'ewc lam=100 c=0.1', 'ewc lam=100 c=0.9', 'ewc lam=100 no clamp']
    nm = ['EWC, λ=10, c=0.5', 'EWC, λ=100, c=0.5', 'EWC, λ=1000, c=0.5', 'EWC, λ=100, c=0.1', 'EWC, λ=100, c=0.9', 'EWC, λ=100, no clamp']
    rows = [['Update rule', 'Loss on Stage-1 memory', 'Loss on new tasks']]
    for k, lab in [('frozen', 'Frozen policy'), ('naive', 'Naive fine-tuning'), ('replay', 'Rehearsal (new + stored data)')]:
        rows.append([lab, f"{R_[k]['ret_mean']:.2f} ± {R_[k]['ret_sd']:.2f}", f"{R_[k]['plas_mean']:.2f}"])
    for k, lab in zip(sw, nm):
        v = R_[k]; rows.append([lab, ('diverged (NaN, 10/10)' if v['finite'] < 10 else f"{v['ret_mean']:.2f} ± {v['ret_sd']:.2f}"), f"{v['plas_mean']:.2f}" if v['finite'] == 10 else 'n/a'])
    x += table('**Table 6.** Update rules over ten stream orders (mean ± s.d.). *c* is the stability clamp *ηλ*·max *F* ≤ *c*.', rows, [3000, 2200, 1500], sz=16)
    x += p1a('The sweep shows that the clamp matters more than λ: without it the penalty step overshoots and the policy diverges in every order, while λ between 10 and 1000 gives the same result because the clamp caps the effective strength. '
             'The trade-off is plain: rehearsal retains most but fits new tasks less well (loss ' + f"{R_['replay']['plas_mean']:.2f}" + ' versus ' + f"{R_['naive']['plas_mean']:.2f}" + ' for naive updating), whereas EWC sits between the two.')
    rt = N.get('retention')
    if rt:
        x += p('Does forgetting hurt search? Re-solving the 400 training tasks at budget 250 with the post-stream policies (stream order 0) gives ' + ', '.join(f"{k} {v['solved']}" for k, v in rt.items()) + ' tasks solved. ' + ('The differences are small because the policy is mixed with a uniform distribution and shallow programs are found regardless of ordering; the loss of Stage-1 knowledge is real in likelihood terms but, at this scale, nearly free in solved tasks.' if max(v['solved'] for v in rt.values()) - min(v['solved'] for v in rt.values()) <= 3 else 'The differences show that forgetting has a direct cost in solved tasks.'))
    return x

def edge_text():
    k = H; v = N[k]
    x = h2('Edge profile')
    h = N['v4_b250']
    x += p1a(f"On two CPU cores the full system with Stage 1b needs a median of {h['sec_median']:.1f} s per task (95th percentile {h['sec_p95']:.1f} s, mean {h['sec_mean']:.1f} s) at budget 250, i.e. about {h['sec_mean'] * 400 / 2 / 60:.0f} minutes for all 400 tasks, and a single search process of the v2 pipeline peaks at about 83 MB of resident memory (measured on 30 training tasks; Stage 1b adds only small lookup tables and was not profiled separately). "
             'There is no GPU, no model download and no network access at inference time, which is the property that matters for ubiquitous deployments. Latency was measured with both cores busy, so single-task latency on an idle core is lower.')
    return x

def faq():
    d = N.get('paired', {})
    ws = [json.load(open(f'/home/claude/results_v2/{f}.json'))['_meta']['wall'] for f in ['eval_v2_S1_nolam', 'eval_v2_H', 'eval_v2_H_S2S3', 'eval_v2_full']]
    WALL = f'{100 * (max(ws) - min(ws)) / min(ws):.0f}%'
    x = h1('Questions a Skeptical Reader Should Ask')
    qa = [
     ('Is the gain simply a bigger library?', f"Largely, and we say so. Moving from 65 to 97 primitives raises the solved count from {cnt('v1_S1')} to {cnt('v2_nolam')} at half the budget, and every v1-solved task remains solved. Stages 2 to 4 leave the solved set unchanged. Rule induction adds three more tasks ({cnt('v2_H_b250')} to {cnt('v3_ind_b250')}) and 19 object-level primitives add six more ({cnt('v4_b250')}), none lost; the second step is significant by a sign test, the first is not. The contribution is the measured decomposition, not a learned-component win."),
     ('Did you tune on the test data?', 'No evaluation task was inspected while writing the primitives; the heuristic weight and depth were tuned on training tasks, and each configuration was evaluated once. The residual risk is that the authors\' familiarity with ARC leaks into the library; Table 4 gives independent checks.'),
     ('Does the verification gate let wrong programs through?', f"Of {g(H, 'verified')} programs verified on the demonstrations at the headline setting, {g(H, 'wrong_verified')} fail on the test input (precision {prec(H)}); Sect. {S.SV} inspects them. Since only verified programs are ever learned from, these are the only source of label-free error."),
     ('Why not an LLM, a transformer proposer, MCTS or an edge board?', 'GPU-scale test-time training reaches far higher ARC accuracy [4]; we study the complementary no-accelerator setting and do not compete with it. The other items were not run: a transformer proposer needs a GPU; the budget curve (Fig. 3) and the beam experiment show that search is not the limit; and the edge profile covers two cloud CPU cores only, not Raspberry Pi or Jetson hardware. Table 7 lists what is measured.'),
    ]
    for q, a in qa:
        x += para(f'**{q}** {a}', 'p1a' if q == qa[0][0] else None)
    return x

def limits_conclusion():
    x = h1('Limitations and Conclusion')
    x += p1a('**Limitations.** Absolute accuracy is low compared with GPU systems; the library is hand-written and the unsolved majority of tasks needs object-level reasoning, counting or drawing that no current primitive expresses; the benefit of rule induction rests on three evaluation tasks and is not statistically significant; the transformer proposer, MCTS and edge-board profiling were not run; half of the DSL v3 gain comes from a single primitive; the forgetting study uses a small policy, so its conclusions need not transfer to large networks; every configuration is a single deterministic run; and only the public evaluation sets are used.')
    x += p('**Conclusion.** On two CPU cores, a verification-gated loop of search, learning, retrieval and consolidation solves '
           + f"{cnt(H)} of 400 ARC-AGI-1 evaluation tasks ({pct(H)}%), and each stage has been priced by an ablation. The primitive library carries most of the gain; rule induction and 19 object-level primitives add a small increment, the second significant by a sign test; the learned prior, retrieval, expert iteration, a partial-credit heuristic, library learning and deeper beam search do not change the solved set; EWC reduces forgetting of the policy by about {N['replay_paired']['ewc_reduction_pct']:.0f}% while rehearsal removes it. "
           'The negative results suggest where effort on edge-class abstract reasoning is best spent: richer, learnable libraries first, and consolidation only with rehearsal. Next, a transformer should propose the programs, evaluated on GPU and edge hardware under the same protocol; transfer to media-specific tasks is untested. Search is deterministic and every number comes from scripts that write per-task logs; code and logs are released with the paper.')
    x += para('**Acknowledgments.** [Funding and acknowledgments to be added.] **Disclosure of Interests.** The authors have no competing interests to declare that are relevant to the content of this article.', 'acknowlegments')
    return x

# ======================================================================== new sections (S1b, beam, examples, extensions)
def induction_text():
    T = N['train_v3']; pi = T['paired_ind']; pn = T['paired_nogate']; sd = pv('v3_ind_b250', 'v2_H_b250'); ind = N.get('v3_induced_solved', []); unc = N.get('v3_induced_unchecked', [])
    tx = N.get('taxonomy', {}); ss = tx.get('same size', [0, 0])
    x = h2('Rule induction for same-size tasks (Stage 1b)')
    rows = [['Setting (budget 250 unless noted)', 'Train solved', 'Train verified', 'Eval solved', 'Eval verified'],
            ['No induction (v2 heuristic)', str(T['noind']['solved']), str(T['noind']['verified']), str(cnt('v2_H_b250')), str(g('v2_H_b250', 'verified'))],
            ['S1b with LOO gate', str(T['ind']['solved']), str(T['ind']['verified']), str(cnt('v3_ind_b250')), str(g('v3_ind_b250', 'verified'))],
            ['S1b without gate', str(T['nogate']['solved']), str(T['nogate']['verified']), str(cnt('v3_ind_b250_nogate')), str(g('v3_ind_b250_nogate', 'verified'))]]
    x += table('**Table 3.** Effect of rule induction. The training columns are not independent evidence, because the feature sets were developed on training tasks; the evaluation columns come from one run per configuration.', rows, [2900, 900, 1000, 900, 1000], sz=15)
    x += p1a(f"Stage 1b is the one addition that moves the solved set. On the 400 training tasks it adds {pi['only_first']} tasks and loses none ({T['noind']['solved']} to {T['ind']['solved']}; sign test p={pi['p']:.3f}), but this is the data on which the feature sets were developed. "
             f"On the evaluation tasks, which were run once, it adds {N['paired']['v3_ind_b250 vs v2_H_b250']['only_first']} tasks and loses none ({cnt('v2_H_b250')} to {cnt('v3_ind_b250')}, {sd}), and the full system reaches {cnt('v3_full')} of 400 ({pct('v3_full')}%, 95% CI {ci('v3_full')}). "
             f"Three tasks are too few to separate from noise: the sign test is not significant (p={N['paired']['v3_ind_b250 vs v2_H_b250']['p']:.2f}), and the interval of the new count overlaps that of the old one. We therefore report the direction (more tasks, none lost, on both sets) and not a proven gain.")
    x += p("The LOO gate did not pay for itself in solved tasks: without it the evaluation count is " + f"{cnt('v3_ind_b250_nogate')} against {cnt('v3_ind_b250')} (one task differs, {pv('v3_ind_b250', 'v3_ind_b250_nogate')}), and precision is {prec('v3_ind_b250')} with the gate and {prec('v3_ind_b250_nogate')} without. "
           + (f"Of the {len(ind)} induced rules that solve an evaluation task ({', '.join(n.replace('induce:', '') for _, n in ind)}), {len(unc)} come from simple feature sets that failed the LOO check and were admitted only because they fill an empty attempt slot; the check ranks candidates, and it did not prevent a wrong answer here. " if ind else '')
           + f"Same-size tasks, which make up {100 * ss[1] / 400:.0f}% of the set, rise from 6 to {ss[0]} solved of {ss[1]} ({100 * ss[0] / max(ss[1], 1):.1f}%). The gain is real but small: induction captures rules that recolour or fill cells by local context, parity, or object size, whereas most same-size tasks move objects, count, or draw, which no lookup of per-cell features expresses.")
    return x

def beam_text():
    b3 = N.get('v3_beam_d3'); b5 = N.get('v3_beam_d5')
    if not b3: return ''
    x = ''
    t = f"We also tried searching deeper rather than longer. We replace best-first search by a beam search of width 100 (state de-duplicated, same heuristic, depth 3 and 5; budget counted in expansions per level) and run the evaluation set once per depth. "
    t += f"At depth 3 the beam solves {b3['solved']} tasks ({b3['verified']} verified), the same {b3['solved']} as best-first search ({pv('v3_beam_d3', 'v2_H')}), in {N['v3_beam_d3']['sec_mean']:.1f} s per task. "
    if b5:
        d = b5['solved'] - b3['solved']
        t += f"At depth 5 it solves {b5['solved']} tasks ({b5['verified']} verified; {pv('v3_beam_d5', 'v3_beam_d3')} against depth 3) at {b5['sec_mean']:.1f} s per task. "
        t += ('Programs of four and five primitives therefore add nothing that the library can use: the unsolved tasks are not "a few steps too deep" but outside the library. ' if d <= 0 else f'Deeper programs add {d} task(s), so part of the unsolved set was a matter of depth, although the gain is small relative to the cost in time. ')
    t += 'We did not implement Monte-Carlo tree search or bidirectional search: the former needs a value estimate that our partial-credit heuristic does not provide beyond what the beam already uses, and the latter needs invertible primitives, whereas most of ours (crops, deduplication, colour removal) are many-to-one.'
    return x + p(t)

def examples_text():
    x = figure('/home/claude/fig_examples.png', '**Fig. 5.** Worked examples (first demonstration, test input, our top prediction; programs found at budget 250 without learned components). Rows 1–3 are solved; row 4 reproduces its demonstrations yet fails on the test input.', 0.8)
    return x + p1a("Fig. 5 makes the mechanism concrete. In row 1 the demonstration inputs are blank grids whose output depends only on the parity of the cell position; Stage 1b fits that table directly and applies it to a larger test grid. "
                   "Row 2 is the two-step program *mirror horizontally, then vertically*, and row 3 a logical OR of the two halves of a split grid followed by a learned colour map, both found by enumeration. "
                   "Row 4 shows the characteristic failure: the program takes the most frequent colour and enlarges it threefold, which fits every demonstration but not the test input.")

def extensions_text():
    mac = N.get('v2_macros', {}); b3 = N.get('v3_beam_d3'); b5 = N.get('v3_beam_d5')
    bt = (f"measured: {b3['solved']} (depth 3){', ' + str(b5['solved']) + ' (depth 5)' if b5 else ''}, none beyond best-first" if b3 and (not b5 or b5['solved'] <= b3['solved']) else (f"measured: {b3['solved']} (depth 3), {b5['solved']} (depth 5)" if b3 else 'n/a'))
    rows = [['Extension', 'Status', 'Result on ARC-AGI-1 evaluation'],
            ['Library learning from verified programs', 'implemented', f"{mac.get('solved', 'n/a')} vs {cnt('v2_H')}: no gain"],
            ['Transformer or GNN program proposer', 'not evaluated', 'GPU script written but untested'],
            ['Beam search (width 100)', 'implemented', bt],
            ['MCTS, bidirectional search', 'not implemented', 'see text; most primitives are not invertible'],
            ['Edge-device profiling (Raspberry Pi, Jetson)', 'not run', 'measured on two cloud CPU cores only']]
    x = h2('Extensions: what was tried and what was not')
    x += table('**Table 7.** Extensions considered, with their status. Only rows marked implemented have measured results in this paper.', rows, [2900, 1300, 2600], sz=15, align=['l', 'l', 'l'])
    x += p1a('The table separates what was measured from what was only planned. Only the extensions that enlarge what the system can express (rule induction and the object-level primitives) produced new solved tasks; the neural proposer for programs outside the hand-written library is the most plausible route to a larger gain.')
    return x

def primitives_text():
    P = N['v4_new_prims']; pi = pv('v4_b250', 'v3_ind_b250'); p2 = pv('v4_b250', 'v2_H_b250'); T = N['train_v4']
    x = h2('Object-level primitives (DSL v3)')
    x += p1a(f"The decomposition says that expressiveness limits the system, so we tested that directly by adding 19 object-level primitives (Sect. 3.1) to the library and re-running with induction at the same budget of 250 expansions. "
             f"The evaluation count rises from {cnt('v3_ind_b250')} to {cnt('v4_b250')} of 400 ({pct('v4_b250')}%, 95% CI {ci('v4_b250')}), with {N['paired']['v4_b250 vs v3_ind_b250']['only_first']} tasks gained and none lost ({pi}); against the v2 heuristic-only system the gain is {N['paired']['v4_b250 vs v2_H_b250']['only_first']} tasks ({p2}). "
             f"On the training tasks it adds {T['paired']['only_first']} and loses {T['paired']['only_second']} ({T['solved']} solved; Table 3). The new solved evaluation tasks use " + ', '.join(f'{k} ({v})' for k, v in sorted(P.items(), key=lambda kv: -kv[1])) + '. '
             "Half of the gain comes from one primitive, foreground-background inversion, so the result is better read as evidence that a few missing primitives matter than as a broad effect; the counts are small and the evaluation set was run once. "
             "The primitives were chosen from general knowledge of ARC transformations before any DSL v3 result was seen, but because the library is hand-written, each such addition also moves the system closer to the benchmark, and the absolute number should be read with that in mind.")
    return x
