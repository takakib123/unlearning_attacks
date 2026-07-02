"""
task4_preeval.py
================
Task 4: full probabilistic pre-eval (n=128) on Q_F and Q_held for ONE model.
Run base and unlearned as two processes on separate GPUs.

  python task4_preeval.py --model locuslab/tofu_ft_llama2-7b --device cuda:0 --tag base
  python task4_preeval.py --model OPTML-Group/SimNPO-TOFU-forget05-Llama-2-7b-chat \
      --device cuda:1 --tag unlearned

Outputs (under experiment_<date>/):
  task4_pre_<tag>_qf.csv  task4_pre_<tag>_qheld.csv  + printed aggregates.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.grpo_core import build_prompt_encodings
from tofu_forget05 import load_forget05, split
from tofu_oracle import load_keywords
from tofu_eval import evaluate_set, aggregate, write_csv, print_agg

TOK = "meta-llama/Llama-2-7b-chat-hf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--keywords_csv", default="tofu_forget05_keywords_DRAFT.csv")
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_level", default="question")
    ap.add_argument("--limit", type=int, default=0, help="cap questions/set for timing")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or f"experiment_{date.today().isoformat()}"
    os.makedirs(out_dir, exist_ok=True)

    kw = load_keywords(args.keywords_csv)
    items = load_forget05()
    for it in items:
        it.keywords = kw.get(it.idx, [])
    sp = split(items, pool_frac=0.25, seed=args.seed, split_level=args.split_level)
    qf, qh = sp.q_f, sp.q_held
    if args.limit:
        qf, qh = qf[:args.limit], qh[:args.limit]

    tok = AutoTokenizer.from_pretrained(TOK)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print(f"[{args.tag}] loading {args.model} rev={args.revision} on {args.device}",
          flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, use_safetensors=True)
    model.to(args.device).eval()

    enc_qf = build_prompt_encodings(tok, qf, "[INST] ", " [/INST]", args.device)
    enc_qh = build_prompt_encodings(tok, qh, "[INST] ", " [/INST]", args.device)

    gk = dict(max_new_tokens=128, temperature=1.0, top_p=0.9, eval_batch=args.eval_batch)
    print(f"[{args.tag}] eval Q_F ({len(enc_qf)}q, n={args.n_eval}) ...", flush=True)
    res_qf = evaluate_set(model, tok, enc_qf, args.n_eval, label=f"{args.tag} Q_F", **gk)
    print(f"[{args.tag}] eval Q_held ({len(enc_qh)}q, n={args.n_eval}) ...", flush=True)
    res_qh = evaluate_set(model, tok, enc_qh, args.n_eval, label=f"{args.tag} Q_held", **gk)

    p_qf = os.path.join(out_dir, f"task4_pre_{args.tag}_qf.csv")
    p_qh = os.path.join(out_dir, f"task4_pre_{args.tag}_qheld.csv")
    write_csv(p_qf, res_qf)
    write_csv(p_qh, res_qh)

    print(f"\n===== TASK 4 PRE-EVAL AGGREGATES [{args.tag}] =====", flush=True)
    print_agg(f"{args.tag} Q_F", aggregate(res_qf))
    print_agg(f"{args.tag} Q_held", aggregate(res_qh))
    print(f"CSVs: {p_qf} | {p_qh}", flush=True)


if __name__ == "__main__":
    main()
