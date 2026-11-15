# CrysFluid-ARC: code, logs and per-task results

Hardware used: 2 CPU cores, 7 GB RAM, no GPU. Python 3.13, numpy, scipy, matplotlib (no torch needed for the CPU system).
Data: git clone https://github.com/fchollet/ARC-AGI, arcprize/ARC-AGI-2, victorvikram/ConceptARC into /home/claude/data/.

Order of execution (all runs write per-task JSON logs and are cached):
1. run_all.py            DSL v1 pipeline (ARC_DSL=v1, default), results/
2. run_replay.py         v1 forgetting replay (superseded by run_replay_v2.py)
3. run_v2.py             DSL v2: tuning on a training subset, ablations, other datasets, scaling, leave-one-family-out, results_v2/
4. run_replay_v2.py      10-order forgetting study (EWC / naive / rehearsal / clamp sweep)
5. run_macros.py make|eval   library learning (macro-primitives)
6. run_retention.py      effect of forgetting on re-solving training tasks
7. run_v1_other.py       DSL v1 on ARC-AGI-2 and ConceptARC
8. analyze.py            tables, confidence intervals, sign tests, figures -> numbers.json
9. make_paper.py         builds the Springer LNCS .docx from numbers.json (needs the splnproc2311 template unpacked at /tmp/tpl)

gpu_arc_ttt.py is an UNTESTED GPU variant (LoRA + test-time training) and is not used for any number in the paper.
