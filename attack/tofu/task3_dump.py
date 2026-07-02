"""
task3_dump.py
=============
Task 3: completion dump + oracle sanity gates.

For a sample of Q_F and Q_held questions, on BOTH the base (pre-unlearning) and the
GradDiff unlearned (checkpoint-30) model, dump greedy + N sampled completions with
the canonical answer, keyword(s), keyword-hit bool, ROUGE-L recall, and the leak
verdict. Then check two sanity gates:

  GATE 1  base leakage > unlearned leakage        (oracle detects signal AND
                                                    unlearning did something)
  GATE 2  unlearned: probabilistic leak > greedy   (the Scholten phenomenon —
                                                    sampling leaks more than mode)
"""
from __future__ import annotations

import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tofu_forget05 import load_forget05, split
from tofu_oracle import load_keywords, leak_oracle

BASE = "locuslab/tofu_ft_llama2-7b"
UNLEARNED = "OPTML-Group/SimNPO-TOFU-forget05-Llama-2-7b-chat"
UNLEARNED_REV = None  # SimNPO repo is single-branch (main)
TOK = "meta-llama/Llama-2-7b-chat-hf"


def load_model(name, device, revision=None):
    m = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=torch.bfloat16, use_safetensors=True)
    m.to(device).eval()
    return m


def gen(model, tok, prompt, device, n, greedy, max_new=128):
    enc = tok(prompt, return_tensors="pt").to(device)
    plen = enc["input_ids"].shape[1]
    if greedy:
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        return [tok.decode(out[0, plen:], skip_special_tokens=True)]
    out = model.generate(
        input_ids=enc["input_ids"].expand(n, -1).contiguous(),
        attention_mask=enc["attention_mask"].expand(n, -1).contiguous(),
        max_new_tokens=max_new, do_sample=True, temperature=1.0, top_p=0.9,
        pad_token_id=tok.pad_token_id)
    return [tok.decode(o[plen:], skip_special_tokens=True) for o in out]


def pick(items, k, seed):
    rng = random.Random(seed)
    return sorted(rng.sample(items, min(k, len(items))), key=lambda it: it.idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keywords_csv", default="tofu_forget05_keywords_DRAFT.csv")
    ap.add_argument("--n_sample", type=int, default=5)
    ap.add_argument("--k_each", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unlearned", default=UNLEARNED)
    ap.add_argument("--unlearned_rev", default=UNLEARNED_REV)
    ap.add_argument("--out", default="task3_completion_dump_simnpo.txt")
    args = ap.parse_args()

    kw_map = load_keywords(args.keywords_csv)
    items = load_forget05()
    for it in items:
        it.keywords = kw_map.get(it.idx, [])
    sp = split(items, pool_frac=0.25, seed=args.seed, split_level="question")
    qf = pick(sp.q_f, args.k_each, args.seed)
    qh = pick(sp.q_held, args.k_each, args.seed + 1)
    sample = [("Q_F", it) for it in qf] + [("Q_held", it) for it in qh]

    tok = AutoTokenizer.from_pretrained(TOK)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print(f"Loading base on cuda:0 and unlearned ({args.unlearned} "
          f"rev={args.unlearned_rev}) on cuda:1 ...")
    base = load_model(BASE, "cuda:0")
    unl = load_model(args.unlearned, "cuda:1", revision=args.unlearned_rev)
    models = {"BASE": (base, "cuda:0"), "UNL": (unl, "cuda:1")}

    # gate accumulators: leak counts
    # Per-question records: greedy leak (0/1), per-sample leaks, any-sample leak.
    # The Scholten mode/mean separation is a PER-QUESTION statistic: does a sample
    # leak where greedy doesn't. Averaging per-completion (mean sample-leak rate)
    # is NOT the phenomenon and can point the wrong way, so we track both but gate
    # on the per-question any-sample-leak fraction.
    rec = {m: [] for m in models}  # list of {"g":int, "any":int, "srate":float}

    f = open(args.out, "w")

    def w(s=""):
        f.write(s + "\n")

    w(f"TASK 3 COMPLETION DUMP — base vs unlearned ({args.unlearned} "
      f"rev={args.unlearned_rev})")
    w(f"oracle: keyword OR ROUGE-L recall>=0.5 | n_sample={args.n_sample} "
      f"temp=1.0 top_p=0.9 | keywords={args.keywords_csv}")
    w("=" * 90)

    for tag, it in sample:
        prompt = f"[INST] {it.question} [/INST]"
        w(f"\n[{tag} q{it.idx}] author={it.author}")
        w(f"  Q: {it.question}")
        w(f"  GOLD: {it.answer}")
        w(f"  keywords: {it.keywords}")
        for mname, (model, dev) in models.items():
            greedy = gen(model, tok, prompt, dev, args.n_sample, greedy=True)[0]
            samples = gen(model, tok, prompt, dev, args.n_sample, greedy=False)
            gl, gkw, grr = leak_oracle(greedy, it.keywords, it.answer, it.question)
            w(f"  --- {mname} ---")
            w(f"    GREEDY  leak={int(gl)} kw={int(gkw)} rougeR={grr:.2f} :: {greedy.strip()[:200]}")
            s_leaks = []
            for j, s in enumerate(samples):
                lk, kwh, rr = leak_oracle(s, it.keywords, it.answer, it.question)
                s_leaks.append(int(lk))
                w(f"    samp{j}   leak={int(lk)} kw={int(kwh)} rougeR={rr:.2f} :: {s.strip()[:200]}")
            rec[mname].append({"g": int(gl), "any": int(any(s_leaks)),
                               "srate": sum(s_leaks) / max(1, len(s_leaks))})

    # gates (per-question fractions)
    def qfrac(m, key):
        r = rec[m]
        return sum(x[key] for x in r) / max(1, len(r))

    w("\n" + "=" * 90)
    w("SANITY GATES (per-question fractions over the dump sample)")
    for m in models:
        w(f"  {m}: greedy_leak_qfrac={qfrac(m,'g'):.3f}  "
          f"any_sample_leak_qfrac={qfrac(m,'any'):.3f}  "
          f"mean_sample_leak_rate={qfrac(m,'srate'):.3f}")
    # GATE 1: base leaks more than unlearned (both greedy and any-sample views)
    g1 = qfrac("BASE", "any") > qfrac("UNL", "any")
    g1g = qfrac("BASE", "g") > qfrac("UNL", "g")
    # GATE 2 (Scholten): unlearned probabilistic (any-sample) >= greedy, with a
    # strict phenomenon instance (greedy=0 but a sample leaks) and no reverse.
    unl = rec["UNL"]
    phen = sum(1 for x in unl if x["g"] == 0 and x["any"] == 1)
    reverse = sum(1 for x in unl if x["g"] == 1 and x["any"] == 0)
    g2 = qfrac("UNL", "any") >= qfrac("UNL", "g") and phen > 0 and reverse == 0
    w(f"\n  GATE 1 (base > unlearned leak): any-sample {'PASS' if g1 else 'FAIL'} "
      f"({qfrac('BASE','any'):.3f} vs {qfrac('UNL','any'):.3f}); "
      f"greedy {'PASS' if g1g else 'FAIL'} "
      f"({qfrac('BASE','g'):.3f} vs {qfrac('UNL','g'):.3f})")
    w(f"  GATE 2 (unlearned probabilistic>greedy, Scholten): "
      f"{'PASS' if g2 else 'FAIL'}  any-sample={qfrac('UNL','any'):.3f} "
      f"greedy={qfrac('UNL','g'):.3f}  phenomenon_q={phen}  reverse_q={reverse}")
    f.close()

    # echo gates to stdout
    print(open(args.out).read().split("SANITY GATES")[1])
    print(f"\nFull dump: {args.out}")


if __name__ == "__main__":
    main()
