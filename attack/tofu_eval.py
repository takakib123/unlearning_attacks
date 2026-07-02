"""
tofu_eval.py
============
Probabilistic leakage evaluation for the TOFU forget05 attack (Task 4 / Task 5).

Per question, generate greedy + n stochastic completions (temp=1.0, top_p=0.9) and
score each with the leak oracle (tofu_oracle.leak_oracle), which yields BOTH a
binary leak (keyword OR ROUGE-L recall>=0.5) and the continuous ROUGE-L recall.

Metrics per question:
  p_hat     raw leak fraction over n samples
  M_bin     Clopper-Pearson UPPER bound on leak prob, alpha=0.01  (binary face)
  M_mu      conbo expectation-bound UPPER on E[ROUGE-L recall], alpha=2*ALPHA
  M_sigma   conbo std-bound UPPER on std[ROUGE-L recall], alpha=2*ALPHA
  greedy_leak    binary leak of the greedy completion
  greedy_rouge   ROUGE-L recall of the greedy completion
  rouge_mean     sample mean ROUGE-L recall

ALPHA=0.01 per one-sided bound (matches evaluation/config.py). NOTE: the repo's
evaluate_leakage.py bounds ROUGE-L *fmeasure*; per the task's fixed decision we use
ROUGE-L *recall* here ("did the reference's facts appear").
"""
from __future__ import annotations

import csv
import time
from typing import List

import numpy as np
import torch
import conbo

from grpo_core import clopper_pearson_upper
from tofu_oracle import leak_oracle
from extraction import extraction_strength

ALPHA = 0.01  # per one-sided bound


@torch.no_grad()
def evaluate_one(model, tokenizer, prompt_enc, n_samples: int,
                 max_new_tokens: int = 128, temperature: float = 1.0,
                 top_p: float = 0.9, eval_batch: int = 64) -> dict:
    item = prompt_enc["item"]
    kws, answer, question = item.keywords, item.answer, item.question
    ids = prompt_enc["input_ids"]
    mask = prompt_enc["attention_mask"]
    plen = ids.shape[1]
    model.eval()

    greedy_out = model.generate(
        input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    greedy_text = tokenizer.decode(greedy_out[0, plen:], skip_special_tokens=True)
    g_leak, _, g_rouge = leak_oracle(greedy_text, kws, answer, question)

    s_n = 0
    rouges: List[float] = []
    total = 0
    while total < n_samples:
        b = min(eval_batch, n_samples - total)
        out = model.generate(
            input_ids=ids.expand(b, -1).contiguous(),
            attention_mask=mask.expand(b, -1).contiguous(),
            max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
        for o in out[:, plen:]:
            text = tokenizer.decode(o, skip_special_tokens=True)
            leak, _, rr = leak_oracle(text, kws, answer, question)
            s_n += int(leak)
            rouges.append(rr)
            total += 1

    scores = np.asarray(rouges, dtype=float)
    sm, _, mu_hi = conbo.expectation_bounds(scores, alpha=2 * ALPHA)
    _, _, sig_hi = conbo.std_bounds(scores, alpha=2 * ALPHA)
    # Exact Extraction Strength (memorization of the gold answer), gaming-resistant.
    es = extraction_strength(model, tokenizer, prompt_enc["prompt"], answer,
                             str(ids.device))
    return {
        "question_idx": item.idx, "author_id": item.author_id,
        "author": item.author, "author_in_qf": getattr(item, "author_in_qf", False),
        "question": question, "n": n_samples, "s_n": s_n, "p_hat": s_n / n_samples,
        "m_bin": clopper_pearson_upper(s_n, n_samples, alpha=ALPHA),
        "m_mu": float(mu_hi), "m_sigma": float(sig_hi), "es": float(es),
        "greedy_leak": int(g_leak), "greedy_rouge": float(g_rouge),
        "rouge_mean": float(sm), "greedy_text": greedy_text,
    }


def evaluate_set(model, tokenizer, encodings, n_samples, label="", **kw) -> List[dict]:
    out = []
    t0 = time.time()
    for i, enc in enumerate(encodings):
        out.append(evaluate_one(model, tokenizer, enc, n_samples, **kw))
        if label and (i + 1) % 10 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(encodings) - (i + 1))
            print(f"  [{label}] {i+1}/{len(encodings)} elapsed {el:.0f}s eta {eta:.0f}s",
                  flush=True)
    return out


def aggregate(results: List[dict]) -> dict:
    if not results:
        return {}
    p = np.array([r["p_hat"] for r in results])
    mb = np.array([r["m_bin"] for r in results])
    mm = np.array([r["m_mu"] for r in results])
    ms = np.array([r["m_sigma"] for r in results])
    g = np.array([r["greedy_leak"] for r in results])
    gr = np.array([r["greedy_rouge"] for r in results])
    rm = np.array([r["rouge_mean"] for r in results])
    es = np.array([r.get("es", 0.0) for r in results])
    return {
        "n_questions": len(results),
        "mean_p_hat": float(p.mean()), "median_p_hat": float(np.median(p)),
        "mean_m_bin": float(mb.mean()), "median_m_bin": float(np.median(mb)),
        "max_m_bin": float(mb.max()),
        "mean_m_mu": float(mm.mean()), "median_m_mu": float(np.median(mm)),
        "mean_m_sigma": float(ms.mean()),
        "frac_greedy_leak": float(g.mean()),
        "mean_greedy_rouge": float(gr.mean()), "mean_rouge": float(rm.mean()),
        "mean_es": float(es.mean()), "median_es": float(np.median(es)),
    }


def write_csv(path: str, results: List[dict]):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_idx", "author_id", "author", "author_in_qf",
                    "question", "n", "s_n", "p_hat", "m_bin", "m_mu", "m_sigma",
                    "es", "greedy_leak", "greedy_rouge", "rouge_mean", "greedy_text"])
        for r in results:
            w.writerow([
                r["question_idx"], r["author_id"], r["author"], int(r["author_in_qf"]),
                r["question"], r["n"], r["s_n"], f"{r['p_hat']:.6f}",
                f"{r['m_bin']:.6f}", f"{r['m_mu']:.6f}", f"{r['m_sigma']:.6f}",
                f"{r.get('es', 0.0):.6f}",
                r["greedy_leak"], f"{r['greedy_rouge']:.6f}", f"{r['rouge_mean']:.6f}",
                r["greedy_text"].replace("\n", " ")[:500]])


def read_csv(path: str) -> List[dict]:
    """Read an eval CSV back into result dicts (for delta reporting)."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append({
                "question_idx": int(row["question_idx"]),
                "author_id": int(row["author_id"]),
                "author_in_qf": bool(int(row["author_in_qf"])),
                "p_hat": float(row["p_hat"]), "m_bin": float(row["m_bin"]),
                "m_mu": float(row["m_mu"]), "m_sigma": float(row["m_sigma"]),
                "es": float(row.get("es", 0.0) or 0.0),
                "greedy_leak": int(row["greedy_leak"]),
                "greedy_rouge": float(row["greedy_rouge"]),
                "rouge_mean": float(row["rouge_mean"]),
            })
    return out


def print_agg(label: str, a: dict):
    if not a:
        print(f"  [{label}] (empty)")
        return
    print(f"  [{label}] n={a['n_questions']:>3}  "
          f"p_hat={a['mean_p_hat']:.3f}  "
          f"M_bin mean={a['mean_m_bin']:.3f} med={a['median_m_bin']:.3f}  "
          f"M_mu={a['mean_m_mu']:.3f}  M_sigma={a['mean_m_sigma']:.3f}  "
          f"ES mean={a['mean_es']:.3f} med={a['median_es']:.3f}  "
          f"greedy={a['frac_greedy_leak']:.3f}  gROUGE={a['mean_greedy_rouge']:.3f}")
