"""CrysFluid-ARC (GPU): transformer pipeline on ARC-AGI-1.  *** UNTESTED on GPU: run the smoke test first. ***

Stage 1  crystallized  : LoRA fine-tune a small pretrained LM on the 400 ARC *training* tasks (random D8 + colour-perm augmentation)
Stage 2  meta-learning : same loop trains on leave-one-out "demos -> held-out demo" episodes (learning to adapt in context)
Stage 3  in-context + test-time training (TTT): per eval task, LoRA steps on leave-one-out episodes built from its demos only
Stage 4  selection     : greedy decode under N augmentations, invert, vote, keep top-2 (ARC pass@2)  [no test labels used]
Stage 5  continual     : --mode reset | naive | ewc   (carry adapter across the task stream; EWC penalty vs. Stage-1 weights)
                         forgetting probe = loss on 16 fixed training episodes, logged every 25 tasks

Usage (Windows, PowerShell):
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install transformers peft accelerate numpy
  git clone https://github.com/fchollet/ARC-AGI.git
  python gpu_arc_ttt.py pretrain --arc ARC-AGI/data --out run1 --samples 2000
  python gpu_arc_ttt.py eval --arc ARC-AGI/data --out run1 --mode reset --limit 5        # smoke test
  python gpu_arc_ttt.py eval --arc ARC-AGI/data --out run1 --mode reset                  # full 400 eval tasks (slow)
  python gpu_arc_ttt.py eval --arc ARC-AGI/data --out run1 --mode ewc --limit 100
"""
import argparse, json, glob, os, random, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

MODEL = "Qwen/Qwen2.5-0.5B"
MAXLEN = 4096

# ------------------------------------------------------------------ data / augmentation
def load(split, arc):
    d = {}
    for f in sorted(glob.glob(f"{arc}/{split}/*.json")):
        t = json.load(open(f))
        d[os.path.basename(f)[:-5]] = {
            "train": [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]],
            "test": [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]}
    return d

def aug(g, t, perm):
    g = np.rot90(g, t % 4)
    if t >= 4: g = np.fliplr(g)
    return perm[g]

def inv(g, t, perm):
    g = np.argsort(perm)[g]
    if t >= 4: g = np.fliplr(g)
    return np.rot90(g, -(t % 4))

def rand_perm(rng):
    p = np.arange(10); q = list(range(1, 10)); rng.shuffle(q); p[1:] = q; return p

def g2s(g): return "\n".join("".join(str(int(c)) for c in r) for r in g)

def build(tok, demos, test_in, target=None):
    """segments -> (input_ids, labels). Loss only on demo outputs and the target output."""
    segs = []
    for i, o in demos:
        segs += [("I\n" + g2s(i) + "\nO\n", False), (g2s(o) + "\n", True)]
    segs += [("I\n" + g2s(test_in) + "\nO\n", False)]
    if target is not None: segs += [(g2s(target) + "\n", True)]
    ids, lab = [], []
    for s, tgt in segs:
        x = tok(s, add_special_tokens=False)["input_ids"]
        ids += x; lab += (x if tgt else [-100] * len(x))
    return ids, lab

def episode(tok, pairs, rng, test_pair=None, maxlen=MAXLEN):
    """pairs: list of (in,out). Random held-out target, random augmentation, drop demos until it fits."""
    pairs = list(pairs); rng.shuffle(pairs)
    if test_pair is None:
        if len(pairs) < 2: return None
        test_pair = pairs.pop()
    t = rng.randrange(8); perm = rand_perm(rng)
    A = lambda g: aug(g, t, perm)
    while pairs:
        ids, lab = build(tok, [(A(i), A(o)) for i, o in pairs], A(test_pair[0]), A(test_pair[1]))
        if len(ids) <= maxlen: return ids, lab
        pairs.pop()
    return None

# ------------------------------------------------------------------ model
def get_model(adapter=None):
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
    m.gradient_checkpointing_enable(); m.enable_input_require_grads(); m.config.use_cache = False
    if adapter: m = PeftModel.from_pretrained(m, adapter, is_trainable=True)
    else:
        m = get_peft_model(m, LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in m.parameters():
        if p.requires_grad: p.data = p.data.float()
    return tok, m

def trainable(m): return [p for p in m.parameters() if p.requires_grad]

def loss_of(m, ids, lab):
    x = torch.tensor([ids]).cuda(); y = torch.tensor([lab]).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return m(input_ids=x, labels=y).loss

def fit(m, eps, lr, ga=4, ewc=None):
    m.train(); m.config.use_cache = False
    ps = trainable(m); opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)
    for k, (ids, lab) in enumerate(eps):
        l = loss_of(m, ids, lab)
        if ewc is not None:
            F, A, lam = ewc
            l = l + 0.5 * lam * sum((f * (p - a) ** 2).sum() for f, a, p in zip(F, A, ps))
        (l / ga).backward()
        if (k + 1) % ga == 0 or k == len(eps) - 1:
            torch.nn.utils.clip_grad_norm_(ps, 1.0); opt.step(); opt.zero_grad(set_to_none=True)

def fisher(m, eps):
    ps = trainable(m); F = [torch.zeros_like(p) for p in ps]; m.train()
    for ids, lab in eps:
        m.zero_grad(set_to_none=True); loss_of(m, ids, lab).backward()
        for f, p in zip(F, ps):
            if p.grad is not None: f += p.grad.detach() ** 2 / len(eps)
    m.zero_grad(set_to_none=True)
    return F

@torch.no_grad()
def probe(m, eps):
    m.eval(); return float(np.mean([loss_of(m, i, l).item() for i, l in eps]))

@torch.no_grad()
def predict(m, tok, demos, test_in, n_aug, rng, maxlen=MAXLEN):
    m.eval(); m.config.use_cache = True; votes = {}
    for a in range(n_aug):
        t = 0 if a == 0 else rng.randrange(8); perm = np.arange(10) if a == 0 else rand_perm(rng)
        A = lambda g: aug(g, t, perm)
        d = list(demos)
        while d:
            ids, _ = build(tok, [(A(i), A(o)) for i, o in d], A(test_in))
            if len(ids) + 940 <= maxlen + 940: break
            d.pop()
        if not d: continue
        ids, _ = build(tok, [(A(i), A(o)) for i, o in d], A(test_in))
        if len(ids) > maxlen: continue
        out = m.generate(torch.tensor([ids]).cuda(), max_new_tokens=940, do_sample=False,
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0, len(ids):], skip_special_tokens=True)
        rows = []
        for line in txt.split("\n"):
            line = line.strip()
            if not line: break
            if not line.isdigit(): rows = None; break
            rows.append([int(c) for c in line])
        if not rows or len({len(r) for r in rows}) != 1: continue
        g = inv(np.array(rows), t, perm); key = json.dumps(g.tolist())
        votes[key] = votes.get(key, 0) + 1
    m.config.use_cache = False
    return sorted(votes.items(), key=lambda kv: -kv[1])[:2]

# ------------------------------------------------------------------ commands
def cmd_pretrain(a):
    rng = random.Random(0); tok, m = get_model(); tr = load("training", a.arc); names = list(tr)
    eps = []
    while len(eps) < a.samples:
        t = tr[rng.choice(names)]; e = episode(tok, t["train"] + t["test"], rng)
        if e: eps.append(e)
    t0 = time.time(); fit(m, eps, lr=2e-4, ga=8)
    m.save_pretrained(f"{a.out}/adapter0"); print("stage1/2 done", round(time.time() - t0), "s")

def cmd_eval(a):
    rng = random.Random(1); tok, m = get_model(f"{a.out}/adapter0")
    tr = load("training", a.arc); ev = load("evaluation", a.arc); names = list(ev); random.Random(0).shuffle(names)
    names = names[:a.limit] if a.limit else names
    prng = random.Random(123); tn = list(tr); probe_eps = []
    while len(probe_eps) < 16:
        t = tr[prng.choice(tn)]; e = episode(tok, t["train"] + t["test"], prng)
        if e: probe_eps.append(e)
    ps = trainable(m); init = [p.detach().clone() for p in ps]; ewc = None
    if a.mode == "ewc":
        fis_eps = []
        while len(fis_eps) < 32:
            t = tr[rng.choice(tn)]; e = episode(tok, t["train"] + t["test"], rng)
            if e: fis_eps.append(e)
        ewc = (fisher(m, fis_eps), init, a.lam)
    base = probe(m, probe_eps); log = open(f"{a.out}/eval_{a.mode}.jsonl", "w"); solved = 0
    print("probe loss before stream:", base)
    for k, n in enumerate(names):
        t = ev[n]; t0 = time.time()
        if a.mode == "reset":
            with torch.no_grad():
                for p, q in zip(ps, init): p.copy_(q)
        eps = []
        for _ in range(a.ttt_steps * 2):
            e = episode(tok, t["train"], rng)
            if e: eps.append(e)
            if len(eps) >= a.ttt_steps: break
        if eps and a.ttt_steps > 0: fit(m, eps, lr=a.ttt_lr, ga=2, ewc=ewc)
        ok = True; preds = []
        for ti, to in t["test"]:
            top = predict(m, tok, t["train"], ti, a.n_aug, rng)
            hit = any(np.array_equal(np.array(json.loads(k_)), to) for k_, _ in top)
            ok &= hit; preds.append([v for _, v in top])
        solved += ok
        rec = {"task": n, "solved": ok, "votes": preds, "sec": round(time.time() - t0, 1)}
        if (k + 1) % 25 == 0: rec["probe_loss"] = probe(m, probe_eps); rec["probe_base"] = base
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(k + 1, "/", len(names), "solved", solved, rec["sec"], "s", flush=True)
    print(f"RESULT mode={a.mode} solved {solved}/{len(names)} = {100 * solved / len(names):.2f}%  probe {base:.4f} -> {probe(m, probe_eps):.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["pretrain", "eval"])
    ap.add_argument("--arc", default="ARC-AGI/data"); ap.add_argument("--out", default="run1")
    ap.add_argument("--samples", type=int, default=2000); ap.add_argument("--mode", default="reset", choices=["reset", "naive", "ewc"])
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--n_aug", type=int, default=4)
    ap.add_argument("--ttt_steps", type=int, default=24); ap.add_argument("--ttt_lr", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=1000.0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    {"pretrain": cmd_pretrain, "eval": cmd_eval}[a.cmd](a)
