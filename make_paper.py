import json, sys
import docxlib as D
from docxlib import *
import paper_static as S
import build_paper as B
N = B.N
def abstract():
    H = B.H; rp = N.get('replay', {}); rpp = N.get('replay_paired', {})
    return (f"Ubiquitous devices must adapt to new situations from a handful of examples, yet the strongest few-shot reasoning systems need GPU-scale hardware. "
            f"We study how far a closed loop of search, learning, retrieval, self-improvement and consolidation can go on two commodity CPU cores, using the Abstraction and Reasoning Corpus (ARC). "
            f"Our five-stage pipeline, CrysFluid-ARC, combines a 116-primitive program library with best-first search, a rule-induction branch for same-size tasks, a meta-learned search policy, retrieval of solved tasks, verification-gated expert iteration (a program may be learned from only if it reproduces every demonstration), "
            f"and online policy updates with stability-clamped Elastic Weight Consolidation (EWC). "
            f"The full system solves {B.cnt(H)} of 400 ARC-AGI-1 evaluation tasks ({B.pct(H)}%, 95% CI {B.ci(H)}), up from {B.cnt('v1_S1')} for its first library, needs about 83 MB of memory, and runs a full pass in minutes without a GPU. "
            f"Controlled ablations give a candid decomposition: the larger library accounts for most of the gain, a new demonstration-driven rule-induction stage adds {B.cnt('v3_ind_b250') - B.cnt('v2_H_b250')} tasks and 19 object-level primitives add {B.cnt('v4_b250') - B.cnt('v3_ind_b250')} more, with no task lost, whereas the learned prior, retrieval, expert iteration, a partial-credit heuristic, library learning and deeper beam search leave the solved set unchanged; "
            f"budget scaling, transfer to ARC-AGI-2 and ConceptARC, and a verification-precision study ({B.prec(H)}) answer a skeptical reader's questions. "
            + (f"Across ten task-stream orders, naive online updating raises the policy's loss on its original knowledge from {rp['frozen']['ret_mean']:.2f} to {rp['naive']['ret_mean']:.2f}; EWC reduces the increase by {rpp['ewc_reduction_pct']:.0f}% and rehearsal removes it ({rp['replay']['ret_mean']:.2f}). " if rp else "")
            + "We release code and logs.")
body = D.front(S.TITLE, S.AUTHORS, S.AFFIL, abstract(), 'abstract reasoning, ARC, test-time adaptation, continual learning, edge intelligence, program search')
body += S.intro(N) + S.related() + S.method(N) + S.protocol(N)
res = B.results() + B.induction_text() + B.primitives_text()
res += h2('Search budget scaling') + B.fig_scaling_text()
res += h2('Which primitive families matter') + B.loco_text()
sc = N.get('stats_concept', {}); a2 = N.get('arc2_eval_v3_full') or N.get('arc2_eval_v2_full', {}); cc = N.get('concept_v3_full') or N.get('concept_v2_full', {})
v1o = (' DSL v1 solves ' + f"{N['arc2_eval_v1']['solved']} of 120 and {N['concept_v1']['solved']} of 160 respectively." ) if ('arc2_eval_v1' in N and 'concept_v1' in N) else ''
res += h2('Transfer to other ARC datasets') + p1a(
    f"Table 4 applies the system, with every learned component fitted on ARC-AGI-1 training tasks only, to ARC-AGI-2 [12] and ConceptARC [11]. On ARC-AGI-2, which was designed to resist library-based search, the full system solves {a2.get('solved')} of 120 tasks, with or without rule induction.{v1o} "
    f"On ConceptARC the full and the base variant both solve {cc.get('solved')} of 160 tasks ({cc.get('acc', 0):.1f}%, CI {cc['ci'][0]:.1f}\u2013{cc['ci'][1]:.1f}%) although {cc.get('verified')} programs reproduce the demonstrations. "
    f"Each ConceptARC task has three test inputs that must all be right, so the per-input view is more informative: the verified programs are correct on {sc.get('inputs_hit_top1')} of {sc.get('inputs_all')} test inputs, "
    f"{100 * sc.get('inputs_hit_top1', 0) / (3 * cc.get('verified', 1)):.0f}% of the {3 * cc.get('verified', 0)} inputs they cover.") + B.datasets_text()
cnts = N.get('solved_depth', {}); wr = N.get('wrong_cases', []); s1 = N.get('stats_arc1', {}); bd = s1.get('by_demos', {}); tx = N.get('taxonomy', {})
def rate(k): return f"{tx[k][0]} of {tx[k][1]}" if k in tx else 'n/a'
res += h2('Verification precision and failure analysis') + p1a(
    f"Of the {B.g(B.H, 'verified')} programs that reproduce all demonstrations, {B.cnt(B.H)} are also correct on the test input (precision {B.prec(B.H)}). "
    f"The {len(wr)} exceptions are overfits, {sum(1 for k, p, m in wr if m)} of them through a learned colour map (for example {wr[0][0]}: {' \u2192 '.join(wr[0][1])} with a {len(wr[0][2])}-entry map). "
    f"With a single attempt (pass@1) the system solves {s1.get('pass1_tasks')} tasks, so the second attempt adds {B.cnt(B.H) - s1.get('pass1_tasks', 0)}; at the level of single test inputs it is right on {s1.get('inputs_hit_top1')} of {s1.get('inputs_all')}. "
    f"Precision depends on how many demonstrations constrain the program: {bd.get('2', [0, 0])[1]} of {bd.get('2', [0, 0])[0]} with two demonstrations, {bd.get('3', [0, 0])[1]} of {bd.get('3', [0, 0])[0]} with three and {bd.get('4+', [0, 0])[1]} of {bd.get('4+', [0, 0])[0]} with four or more; "
    f"the ConceptARC contrast ({sc.get('by_demos', {}).get('2', [0, 0])[1]} of {sc.get('by_demos', {}).get('2', [0, 0])[0]} with two demonstrations) shows that the gate also depends on how far test inputs stray from the demonstrations. "
    f"Solved programs are short: {cnts.get('1', 0)} use one primitive, {cnts.get('2', 0)} two and {cnts.get('3', 0)} three, and {N.get('solved_cmap', 0)} need a learned colour map. "
    f"Every task solved by v1 is also solved by the full system, which adds {N.get('union_v1_v2', {}).get('v2_only')} tasks.") + B.taxonomy_text() + p1a(
    f"Table 5 shows where the solved tasks are: {rate('smaller output')} tasks whose output is smaller than the input (crops, reductions), {rate('larger output')} with a larger output (scaling, tiling, mirroring) and {rate('1x1 output')} single-cell outputs. "
    f"The {tx.get('same size', [0, 0])[1]} same-size tasks, {100 * tx.get('same size', [0, 1])[1] / 400:.0f}% of the set, are almost untouched ({rate('same size')} solved). In-place edits that recolour or move objects according to relations between them are the frontier for a learned proposal model.") + B.examples_text()
res += B.library_text() + B.beam_text() + B.forgetting_text() + B.edge_text() + B.extensions_text()
body += res + B.faq() + B.limits_conclusion() + S.references()
import json as _j, re as _re
def _blocks(x):
    out = []; i = 0
    while i < len(x):
        m = _re.compile(r'<w:(p|tbl)[ >]').search(x, i)
        if not m: out.append(x[i:]); break
        if m.start() > i: out.append(x[i:m.start()])
        tag = m.group(1)
        if tag == 'p':
            j = x.index('</w:p>', m.start()) + 6
        else:
            depth = 0; j = m.start()
            for t in _re.finditer(r'<(/?)w:tbl[ >]', x[m.start():]):
                depth += -1 if t.group(1) else 1
                if depth == 0: j = m.start() + t.end(); j = x.index('</w:tbl>', m.start() + t.start()) + 8; break
        out.append(x[m.start():j]); i = j
    return out
def shift_figures(x, shifts):
    bl = _blocks(x); figs = [i for i, b in enumerate(bl) if '<w:drawing>' in b and 'figurecaption' in bl[i + 1]]
    moves = []
    for k, i in enumerate(figs):
        n = shifts.get(str(k), 0)
        if n: moves.append((i, n))
    # apply from last to first so indices stay valid
    for i, n in sorted(moves, reverse=True):
        pair = [bl[i], bl[i + 1]]; del bl[i:i + 2]
        t = max(0, min(len(bl), i + n)); bl[t:t] = pair
    return ''.join(bl)
import os as _os
body = shift_figures(body, _j.loads(_os.environ.get('FIGSHIFT', '{}')) if not _os.path.exists('/home/claude/figshift.json') else _j.load(open('/home/claude/figshift.json')))
out = '/home/claude/CrysFluid-ARC_Ubi-Media2027.docx'
D.build(body, out, 'A. Anny Leema and P. Balakrishnan', 'CrysFluid-ARC')
print('built', out)
