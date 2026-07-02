"""
add_es.py
=========
Augment existing eval CSVs with the exact Extraction Strength column (one teacher-
forced forward pass per question — cheap, no sampling). Lets us add ES to the Task 4
pre-eval CSVs without re-running the expensive n=128 sampling.

  python add_es.py --model OPTML-Group/SimNPO-TOFU-forget05-Llama-2-7b-chat \
      --device cuda:1 --csvs experiments/experiment_2026-07-01/task4_pre_unlearned_qf.csv \
      experiments/experiment_2026-07-01/task4_pre_unlearned_qheld.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tofu_forget05 import load_forget05
from shared.extraction import extraction_strength

TOK = "meta-llama/Llama-2-7b-chat-hf"
NEW_COLS = ["question_idx", "author_id", "author", "author_in_qf", "question",
            "n", "s_n", "p_hat", "m_bin", "m_mu", "m_sigma", "es",
            "greedy_leak", "greedy_rouge", "rouge_mean", "greedy_text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--csvs", nargs="+", required=True)
    args = ap.parse_args()

    ans = {it.idx: it.answer for it in load_forget05()}
    tok = AutoTokenizer.from_pretrained(TOK)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, use_safetensors=True)
    model.to(args.device).eval()

    for path in args.csvs:
        rows = list(csv.DictReader(open(path, newline="")))
        for r in rows:
            qi = int(r["question_idx"])
            prompt = f"[INST] {r['question']} [/INST]"
            r["es"] = f"{extraction_strength(model, tok, prompt, ans[qi], args.device):.6f}"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(NEW_COLS)
            for r in rows:
                w.writerow([r.get(c, "") for c in NEW_COLS])
        print(f"augmented {path} with ES ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
