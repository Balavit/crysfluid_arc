from docxlib import *

SV, SF, SE = '5.7', '5.9', '5.10'   # verification / forgetting subsections
TITLE = 'CrysFluid-ARC: Verification-Gated Self-Improvement and Forgetting-Aware Consolidation for Few-Shot Abstract Reasoning on Edge-Class CPUs'
AUTHORS = 'A. Anny Leema^1^ and P. Balakrishnan^2,\u2217^'
AFFIL = '^1,2^ School of Computer Science and Engineering, Vellore Institute of Technology, India  \u2217 Corresponding author: P. Balakrishnan, balakrishnan.p@vit.ac.in'

def intro(N):
    x = h1('Introduction')
    x += p1a('Ubiquitous-media systems must adapt to situations that were not anticipated at design time: a new device, a new user, a new media format. '
             'Models trained offline keep their *crystallized* knowledge in fixed parameters and degrade when conditions shift, while purely reactive systems have no foundation to reason from. '
             'The Abstraction and Reasoning Corpus (ARC) [1] isolates the missing capability in its purest form. Every task is new, only two to five demonstration pairs are given, and success means inferring the hidden rule and applying it to an unseen input. '
             'Recent GPU-scale systems based on large language models with test-time training make strong progress on ARC [2,4], but they presuppose datacenter hardware, which is precisely what many ubiquitous deployments lack.')
    x += p('This paper asks a narrower and more answerable question: *how far can a closed loop of search, learning, retrieval, self-improvement and consolidation go on two commodity CPU cores, and what is each component actually worth?* '
           'We organise the loop as a five-stage pipeline, CrysFluid-ARC, that turns crystallized knowledge (a primitive library) into fluid adaptation (search, retrieval) and back into consolidated knowledge (a learned policy that is updated without forgetting). '
           'We deliberately do not claim artificial general intelligence. We claim a set of measured, reproducible findings, each checked by a controlled ablation, and we report negative results where the data show them.')
    x += p('**Contributions.** (1) *Verification-gated self-improvement.* Programs are accepted, retrieved and used as training signal only if they reproduce every demonstration pair, so the system improves itself without any test label; we measure how often this gate is wrong (Sect. ' + SV + '). '
           '(2) *Demonstration-driven rule induction* (Stage 1b): per-cell lookup rules for same-size tasks, checked by a leave-one-demonstration-out gate. '
           '(3) *A reproducible edge-class baseline* on ARC-AGI-1, ARC-AGI-2 and ConceptARC with ablations, budget scaling, confidence intervals and paired significance tests. '
           '(4) *A stability-clamped online EWC update*, compared with naive updating and rehearsal over ten streams: for a small policy network, rehearsal, not EWC, is the effective remedy. '
           '(5) *A candid account of what does not help*: the learned prior, retrieval, a partial-credit heuristic, library learning and deeper search add little once the library is the bottleneck.')
    x += p('**Relevance to ubiquitous media.** Ubiquitous-media services run on phones, wearables and gateways, where a new content format or sensor layout appears with only a few labelled examples and no accelerator is available. '
           'ARC is not a media benchmark and we evaluate no media data; the link is the capability it isolates: inferring a transformation from a few examples, verifying it, and keeping what was learned, within a CPU budget. '
           'Sect. ' + SE + ' reports the footprint, in the spirit of tiny-device benchmarks [23]; learning like people from few examples is the motivation of [26].')
    return x

def related():
    x = h1('Related Work')
    x += p1a('**ARC and program synthesis.** ARC was introduced together with a definition of intelligence as skill-acquisition efficiency [1]. Hand-written domain-specific languages with depth-limited search dominated the 2020 Kaggle competition, and the ARC Prize reports describe how top entries combine program search with learned components [2]. '
             'DreamCoder learns a library and a neural search prior jointly [3], and DeepCoder showed that a learned model can guide enumerative search [13]; our Stages 1, 2 and 4 follow this tradition at small scale, with a verifier built from the demonstrations. Programming by example has a longer history in spreadsheet synthesis [25], and humans solve most ARC tasks by inducing such programs [20].'
             ' Language models are weak when prompted directly [28], and recent systems add explicit hypothesis generation [19] or combine induction with direct prediction [18]; CodeIt [17] iterates program sampling with hindsight replay and solves 15% of the evaluation tasks with a pretrained language model, the closest relative of our self-improvement loop, which we run with search instead of a language model, and STaR [21] applies the same keep-only-verified idea to rationales.')
    x += p('**Test-time training.** Fine-tuning a language model on a task\'s own demonstrations, with augmentation and voting, gives large gains on ARC [4], typically with low-rank adapters [24], and procedural example generators supply training data [14]. These systems need GPUs; we study the complementary CPU-only regime and release a GPU variant for future work. '
           '**Meta-learning, retrieval and expert iteration.** MAML learns an initialisation for fast adaptation [6]; retrieval-augmented generation grounds a model in stored examples [7]; expert iteration alternates search with imitation of its own verified results [8].')
    x += p('**Continual learning.** Elastic Weight Consolidation (EWC) protects parameters important to earlier knowledge through the Fisher information [5]; the problem was named catastrophic interference in [27]; rehearsal replays stored examples [10], and experience replay alone can outperform parameter-regularisation methods in reinforcement learning [22]; reviews are given in [16]. Most evidence comes from image classifiers. Whether a lightweight consolidation rule helps a search-guiding policy that is updated from its own verified successes is, to our knowledge, not reported for ARC. '
           '**Benchmarks.** We evaluate on ARC-AGI-1 [1], ARC-AGI-2 [12] and ConceptARC [11], which tests individual spatial concepts.')
    return x

def method(N):
    d = N.get('decisions') or {}
    lam = d.get('lam', '?'); dep = d.get('depth', '?')
    x = h1('The CrysFluid-ARC Pipeline')
    x += p1a('Fig. 1 shows the loop and Fig. 2 the path of a single task through it, with the offline and online learning loops that run around it; Table 1 summarises the role of each stage. A task supplies demonstration pairs (*x*~i~, *y*~i~) and test inputs. A *program* is a sequence of at most *D* primitives, optionally followed by a colour map fitted on the demonstrations. A program is *verified* if it reproduces every demonstration output exactly; this is the only reward signal in the whole system, and test outputs are never consulted before scoring.')
    x += figure('/home/claude/fig_arch.png', '**Fig. 1.** The five-stage CrysFluid-ARC loop. Stage 1 supplies a primitive library and search; Stages 2 and 3 steer it with a learned policy and retrieved programs; the verifier gates what may be learned from (Stage 4); Stage 5 consolidates the policy during a task stream.', 1.0)
    x += h2('Stage 1: crystallized base and search')
    x += p1a(f'The DSL v2 library holds 97 grid primitives in eight families: geometry (7), object extraction (8), scaling and tiling (15), mirroring and symmetry (10), physics and filling (7), colour (21), grid logic (14; boolean and overlay operations on the two halves of a split grid) and line drawing (15; rays, row/column fills, borders, shifts). '
             'The first 65 form DSL v1; the 32 additions form DSL v2; DSL v3 adds 19 object-level primitives (116 in total): rigid movement of every object in four directions, connecting same-coloured pixels along rows and columns, diagonal rays, selecting a cell of a gridded input (most or fewest filled, or the unique one), colour histograms as a column or as bars, swapping the two main colours, inverting foreground and background, outlining objects, cropping the most colourful object, recolouring objects by size rank, and filling regions enclosed by an object. All three were written from general knowledge of common ARC transformation types, without inspecting any individual evaluation task. Search is best-first over primitive sequences with state de-duplication: states are the tuples of intermediate grids over all demonstration inputs, so programs that behave identically are expanded once. '
             'The priority of a partial program with path cost *g* and current state *s* follows the A*-style form of [9]:')
    x += equation('*\u03c6*(*s*) = *g*(*s*) − *μ* · *h*(*s*),   *g*(*s*) = Σ~t~ −log *p*(*a*~t~ | *f*~task~)', '1')
    x += p1a('where *h*(*s*) in [0,1] is a partial-credit score: the mean pixel accuracy against the demonstration outputs when shapes agree, and otherwise a shape-and-colour-histogram similarity. '
             f'*μ* = 0 recovers plain uniform-cost search. The weight and depth were chosen on a half-size subset of the training tasks (*μ* = {lam}, *D* = {dep}), never on evaluation tasks. Search stops at a node budget or after two distinct test predictions, matching the two attempts allowed by ARC.')
    x += h2('Stage 1b: demonstration-driven rule induction')
    x += p1a('Enumeration over a primitive library cannot express rules such as "recolour every object by its size" or "fill each cell by its row and column parity", which dominate the same-size tasks (68% of the evaluation set). '
             'For tasks whose inputs and outputs share a shape, Stage 1b therefore *induces* a rule directly from the demonstrations. A rule is a lookup table that maps a per-cell feature to an output colour; we try 13 feature sets, from simple to complex: the cell colour *c*, *c* with its 8-neighbour colour count, *c* with the set of 4-neighbour colours, *c* with the size, size rank or rectangularity of its connected component, *c* with the parity of its position, the row or column index, the row-and-column pair, and the raw 4- and 8-neighbourhoods. '
             'A table is a *candidate* only if it is consistent with every demonstration cell, is not the identity, and covers at least 99% of the test cells (an unseen feature value would force a guess). '
             'The seven simplest feature sets are accepted directly; the six that can memorise a demonstration (position- and neighbourhood-based) must in addition pass a leave-one-demonstration-out (LOO) check: the table fitted on all but one demonstration must reproduce the held-out one. '
             'This is the verification gate of Stage 4 applied to induced rules. Candidates fill whichever of the two attempt slots enumeration leaves empty, so induction can never displace a verified program.')
    x += figure('/home/claude/fig_flow.png', width_frac=0.8, caption='**Fig. 2.** What happens to one task at inference time. Branch A enumerates programs (Stages 1\u20133); Branch B induces a rule (Stage 1b). Both pass a demonstration-based gate before entering a shared pool from which at most two distinct predictions are returned. No test output is used.')
    x += h2('Stages 2 and 3: meta-learned prior and retrieval')
    x += p1a('Stage 2 trains a task-conditioned softmax policy *p*(*a* | *f*) = softmax(*Wf* + *b*) on a 19-dimensional task descriptor (size ratios, colour counts, symmetry flags, object counts) by maximum likelihood on verified programs from training tasks, and mixes it 0.7/0.3 with a uniform distribution. '
             'Stage 3 retrieves the five nearest solved tasks (Euclidean distance on standardised descriptors) and verifies their programs on the new demonstrations before any search starts.')
    x += table('**Algorithm 1.** Inference for one task (pass@2). *B*: node budget; *D*: maximum depth; π: policy; *M*: memory.', [
        ['Solve(task, B, D, π, M)'],
        ['1\u00a0\u00a0*seeds* \u2190 programs of the 5 nearest tasks in *M* (standardised descriptor *f*~task~)'],
        ['2\u00a0\u00a0*pool* \u2190 [ *p* in *seeds* if *p* reproduces all demonstration outputs ]'],
        ['3\u00a0\u00a0**while** nodes < *B* **and** |distinct test predictions in *pool*| < 2:'],
        ['4\u00a0\u00a0\u00a0\u00a0\u00a0pop *s* with smallest \u03c6(*s*) = *g*(*s*) \u2212 \u03bc*h*(*s*); skip if its state was seen'],
        ['5\u00a0\u00a0\u00a0\u00a0\u00a0**if** *s* (with a colour map fitted on the demos) matches all demonstration outputs: add to *pool*'],
        ['6\u00a0\u00a0\u00a0\u00a0\u00a0**else if** depth(*s*) < *D*: push *s* + each primitive *a*, with cost *g* \u2212 log \u03c0(*a* | *f*)'],
        ['7\u00a0\u00a0**if** shapes(input) = shapes(output) **and** |pool| < 2:  \u00a0 \u25b7 Stage 1b'],
        ['8\u00a0\u00a0\u00a0\u00a0\u00a0**for** feature set *k* = 1..13: fit table; keep if consistent, non-identity, \u226599% test coverage,'],
        ['\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0and (*k* is simple **or** it passes the LOO check); add to *pool* until it holds two'],
        ['9\u00a0\u00a0**return** the first two distinct test predictions of *pool* (enumerated before induced)'],
    ], [6900], header=1, sz=15, align=['l'])
    x += h2('Stage 4: verification-gated expert iteration')
    x += p1a('The policy and memory are used to search the training tasks again with leave-one-out memory. Newly verified programs enlarge the training set and the memory, and the policy is retrained. A candidate enters the memory only if it passes the demonstration verifier, so noise cannot enter through test labels, but a program that fits the demonstrations by accident can still enter; Sect. ' + SV + ' quantifies this.')
    x += h2('Stage 5: consolidation during a task stream')
    x += p1a('On a stream of unlabeled tasks, the policy is updated online from tasks the verifier accepted. To protect Stage-1 knowledge, we add the EWC penalty [5]')
    x += equation('*L*(*θ*) = *L*~new~(*θ*) + Σ~i~ (*λ*/2) *F*~i~ (*θ*~i~ − *θ*^∗^~i~)^2^', '2')
    x += p1a('with the diagonal empirical Fisher *F* accumulated online. A first version of this rule diverged numerically, so we clamp the penalty so that *ηλ*·max *F* ≤ *c* (*c* = 0.5 by default; Sect. ' + SF + ' sweeps it). We compare against naive updating and against rehearsal, which fits the new data together with the stored Stage-1 data.')
    x += table('**Table 1.** Role of each stage and what we measured for it (ARC-AGI-1 evaluation, details in Sect. 5).', [
        ['Stage', 'Role', 'Learns from', 'Measured effect on solved tasks'],
        ['S1 Library + search', 'proposes programs', 'nothing (hand-written)', f"carries the gain: {N['v1_S1']['solved']} \u2192 {N['v2_nolam']['solved']} (+32 primitives), \u2192 {N['v4_b250']['solved']} (+19 more, with S1b)"],
        ['S1b Rule induction', 'rules for same-size tasks', 'demonstrations of the task', f"{N['v2_H_b250']['solved']} \u2192 {N['v3_ind_b250']['solved']} at budget 250 (Table 3)"],
        ['S2 Policy', 'orders the search', 'verified training programs', 'no change at budget \u2265 250'],
        ['S3 Retrieval', 'seeds search with solved tasks', 'memory of solved tasks', 'no change'],
        ['S4 Expert iteration', 'grows S2/S3 training data', 'own verified programs', 'no change; gate precision in Sect. ' + SV],
        ['S5 Consolidation', 'keeps S2 stable in a stream', 'stream of verified tasks', f"EWC \u2212{N['replay_paired']['ewc_reduction_pct']:.0f}% drift; rehearsal removes it"],
    ], [1450, 1650, 1700, 2000], sz=13, align=['l', 'l', 'l', 'l'])
    return x

def protocol(N):
    x = h1('Experimental Protocol')
    x += p1a('**Data.** ARC-AGI-1: 400 training and 400 public evaluation tasks; ARC-AGI-2: 120 public evaluation tasks; ConceptARC: 160 tasks in 16 concept groups. '
             'All learned components (standardiser, policy, memory) are fitted on ARC-AGI-1 *training* tasks only and then applied unchanged to every other set. '
             '**Metric.** A task is solved if every test output is matched by one of at most two predictions (pass@2, the ARC convention); we report counts with Wilson 95% intervals [15] and exact sign tests on discordant task pairs. '
             '**Hygiene.** Neither DSL was written with an evaluation task in view; the heuristic weight and search depth were tuned on training tasks only; each evaluation configuration is run once; no hyperparameter was selected on solved-task counts of an evaluation set. '
             'One deviation is disclosed: the first EWC run on the evaluation stream diverged numerically, and the stability clamp was added in response (Sect. ' + SF + ' reports the unclamped variant). '
             '**Hardware.** Two CPU cores, 7 GB of RAM, no GPU; search is deterministic and single-threaded per task. '
             '**Baselines.** Identity with colour map (depth 0), single-primitive programs (depth 1), DSL v1 search, and each ablated variant of v2. '
             '**Stage 1b hygiene.** The feature sets of Stage 1b were developed on ARC-AGI-1 training tasks (a 60-task smoke test, then the 400-task training run, which is therefore not independent evidence); the evaluation set was run once with the LOO gate and once without, and both results are reported.')
    return x

REFS = [
 'Chollet, F.: On the Measure of Intelligence. arXiv:1911.01547 (2019)',
 'Chollet, F., Knoop, M., Kamradt, G., Landers, B.: ARC Prize 2024: Technical Report. arXiv:2412.04604 (2024)',
 'Ellis, K., Wong, C., Nye, M., et al.: DreamCoder: Bootstrapping inductive program synthesis with wake-sleep library learning. In: PLDI 2021, pp. 835–850 (2021)',
 'Akyürek, E., Damani, M., Qiu, L., Guo, H., Kim, Y., Andreas, J.: The surprising effectiveness of test-time training for abstract reasoning. arXiv:2411.07279 (2024)',
 'Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al.: Overcoming catastrophic forgetting in neural networks. PNAS 114(13), 3521–3526 (2017)',
 'Finn, C., Abbeel, P., Levine, S.: Model-agnostic meta-learning for fast adaptation of deep networks. In: ICML 2017, pp. 1126–1135 (2017)',
 'Lewis, P., Perez, E., Piktus, A., et al.: Retrieval-augmented generation for knowledge-intensive NLP tasks. In: NeurIPS 2020 (2020)',
 'Anthony, T., Tian, Z., Barber, D.: Thinking fast and slow with deep learning and tree search. In: NeurIPS 2017 (2017)',
 'Hart, P.E., Nilsson, N.J., Raphael, B.: A formal basis for the heuristic determination of minimum cost paths. IEEE Trans. Syst. Sci. Cybern. 4(2), 100–107 (1968)',
 'Robins, A.: Catastrophic forgetting, rehearsal and pseudorehearsal. Connection Science 7(2), 123–146 (1995)',
 'Moskvichev, A., Odouard, V.V., Mitchell, M.: The ConceptARC benchmark: evaluating understanding and generalization in the ARC domain. TMLR (2023)',
 'Chollet, F., Knoop, M., Kamradt, G., Landers, B., Pinkard, H.: ARC-AGI-2: a new challenge for frontier AI reasoning systems. arXiv:2505.11831 (2025)',
 'Balog, M., Gaunt, A.L., Brockschmidt, M., Nowozin, S., Tarlow, D.: DeepCoder: learning to write programs. In: ICLR 2017 (2017)',
 'Hodel, M.: Addressing the abstraction and reasoning corpus via procedural example generation. arXiv:2404.07353 (2024)',
 'Wilson, E.B.: Probable inference, the law of succession, and statistical inference. J. Am. Stat. Assoc. 22(158), 209–212 (1927)',
 'Parisi, G.I., Kemker, R., Part, J.L., Kanan, C., Wermter, S.: Continual lifelong learning with neural networks: a review. Neural Networks 113, 54–71 (2019)',
 'Butt, N., Manczak, B., Wiggers, A., Rainone, C., Zhang, D.W., Defferrard, M., Cohen, T.: CodeIt: self-improving language models with prioritized hindsight replay. In: ICML 2024. arXiv:2402.04858 (2024)',
 'Li, W.-D., Hu, K., Larsen, C., et al.: Combining induction and transduction for abstract reasoning. arXiv:2411.02272 (2024)',
 'Wang, R., Zelikman, E., Poesia, G., Pu, Y., Haber, N., Goodman, N.D.: Hypothesis search: inductive reasoning with language models. In: ICLR 2024 (2024)',
 'Johnson, A., Vong, W.K., Lake, B.M., Gureckis, T.M.: Fast and flexible: human program induction in abstract reasoning tasks. In: CogSci 2021. arXiv:2103.05823 (2021)',
 'Zelikman, E., Wu, Y., Mu, J., Goodman, N.D.: STaR: bootstrapping reasoning with reasoning. In: NeurIPS 2022 (2022)',
 'Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T.P., Wayne, G.: Experience replay for continual learning. In: NeurIPS 2019, pp. 350–360 (2019)',
 'Banbury, C., Reddi, V.J., Torelli, P., et al.: MLPerf Tiny benchmark. In: NeurIPS 2021 Datasets and Benchmarks Track. arXiv:2106.07597 (2021)',
 'Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: low-rank adaptation of large language models. In: ICLR 2022 (2022)',
 'Gulwani, S.: Automating string processing in spreadsheets using input-output examples. In: POPL 2011, pp. 317–330 (2011)',
 'Lake, B.M., Ullman, T.D., Tenenbaum, J.B., Gershman, S.J.: Building machines that learn and think like people. Behav. Brain Sci. 40, e253 (2017)',
 'McCloskey, M., Cohen, N.J.: Catastrophic interference in connectionist networks: the sequential learning problem. Psychol. Learn. Motiv. (1989). https://doi.org/10.1016/S0079-7421(08)60536-8',
 'Xu, Y., Li, W., Vaezipoor, P., Sanner, S., Khalil, E.B.: LLMs and the Abstraction and Reasoning Corpus: successes, failures, and the importance of object-based representations. Trans. Mach. Learn. Res. (2024)'
]
# NOTE: references are numbered in the order above; in-text numbers [1..16] follow this list. Entries 17+ unused.
def references():
    return h1('References') + ''.join(ref(r) for r in REFS)
