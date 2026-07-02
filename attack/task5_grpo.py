"""
task5_grpo.py
=============
Task 5: GRPO relearning attack on the SimNPO forget05 unlearned model, trained on
Q_F ONLY, then post-eval on Q_F (in-set) and Q_held (generalization). Config ported
from the HP attack (grpo_hp_multi_v2 / grpo_core): LoRA r8 on q_proj/v_proj a16,
G=8, KL beta=0.01 (anchored to the unlearned policy), clip 0.2, K=4, lr 1e-4,
skip-saturated (std<1e-3), early stop on rolling-mean reward.

Reward = the leak oracle (keyword OR ROUGE-L recall>=0.5), binary. Degenerate
completions score 0 (tofu_oracle guard), discouraging collapse.

Pre-attack reference = Task 4 unlearned CSVs (a freshly-attached LoRA with B=0 is
identity, so Task 4 unlearned == this script's pre). Post-eval uses the identical
tofu_eval code path.

Run:
  python task5_grpo.py --device cuda:0 \
      --pre_qf experiment_2026-07-01/task4_pre_unlearned_qf.csv \
      --pre_qheld experiment_2026-07-01/task4_pre_unlearned_qheld.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import time
from dataclasses import dataclass
from datetime import date
from typing import Tuple

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from grpo_core import (
    attach_new_lora, build_prompt_encodings, grpo_loss,
    policy_forward_with_kl, sample_rollouts,
)
from tofu_forget05 import load_forget05, split
from tofu_oracle import load_keywords, leak_oracle
from tofu_eval import (
    aggregate, evaluate_set, print_agg, read_csv, write_csv,
)
from extraction import es_rollout

TOK = "meta-llama/Llama-2-7b-chat-hf"
UNLEARNED = "OPTML-Group/SimNPO-TOFU-forget05-Llama-2-7b-chat"


@dataclass
class Cfg:
    group_size: int = 8
    max_new_tokens: int = 128
    sampling_temperature: float = 1.0
    sampling_top_p: float = 0.9


def reward_of(text, item, tokenizer, es_w: float):
    """Blend: (1-w)*binary leak oracle + w*rollout extraction strength. The ES
    term rewards verbatim reproduction of gold tokens (gaming-resistant), while the
    oracle keeps a signal on keyword/ROUGE leakage. w=0 recovers the pure oracle."""
    leak, _, _ = leak_oracle(text, item.keywords, item.answer, item.question)
    leak = 1.0 if leak else 0.0
    if es_w <= 0.0:
        return leak
    esr = es_rollout(text, item.answer, tokenizer)
    return (1.0 - es_w) * leak + es_w * esr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--keywords_csv", default="tofu_forget05_keywords_DRAFT.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_level", default="question")
    # GRPO
    ap.add_argument("--num_outer_steps", type=int, default=150)
    ap.add_argument("--prompts_per_step", type=int, default=8)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--kl_beta", type=float, default=1e-2)
    ap.add_argument("--es_reward_weight", type=float, default=0.5,
                    help="proportion of extraction strength in the reward blend")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--sat_std", type=float, default=1e-3)
    ap.add_argument("--early_stop_window", type=int, default=20)
    ap.add_argument("--early_stop_threshold", type=float, default=0.9)
    # eval
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--pre_qf", required=True)
    ap.add_argument("--pre_qheld", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = args.out_dir or f"experiment_{date.today().isoformat()}"
    os.makedirs(out_dir, exist_ok=True)
    cfg = Cfg()

    kw = load_keywords(args.keywords_csv)
    items = load_forget05()
    for it in items:
        it.keywords = kw.get(it.idx, [])
    sp = split(items, pool_frac=0.25, seed=args.seed, split_level=args.split_level)
    qf, qh = sp.q_f, sp.q_held
    print(f"|Q_F|={len(qf)} |Q_held|={len(qh)}", flush=True)

    tok = AutoTokenizer.from_pretrained(TOK)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print(f"Loading {UNLEARNED} on {args.device} + fresh LoRA r{args.lora_rank}",
          flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        UNLEARNED, dtype=torch.bfloat16, use_safetensors=True)
    base.to(args.device).eval()
    model = attach_new_lora(base, args.lora_rank, args.lora_alpha, 0.0,
                            ("q_proj", "v_proj"))
    model.print_trainable_parameters()

    enc_qf = build_prompt_encodings(tok, qf, "[INST] ", " [/INST]", args.device)
    enc_qh = build_prompt_encodings(tok, qh, "[INST] ", " [/INST]", args.device)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    train_log = open(os.path.join(out_dir, "task5_train_log.csv"), "w", newline="")
    tw = csv.writer(train_log)
    tw.writerow(["step", "wall_s", "mean_reward", "n_saturated", "pg_loss",
                 "kl_loss", "clip_frac", "grad_norm"])

    eff_pps = min(args.prompts_per_step, len(enc_qf))
    rng = np.random.RandomState(args.seed + 99991)
    recent = collections.deque(maxlen=args.early_stop_window)
    t0 = time.time()
    stopped_at, reason = args.num_outer_steps, "max_steps"

    for step in range(args.num_outer_steps):
        sel = rng.choice(len(enc_qf), size=eff_pps, replace=False).tolist()
        buf = []
        for pi in sel:
            enc = enc_qf[pi]
            full_ids, full_mask, comp_mask, old_lp, comps = sample_rollouts(
                model, tok, cfg, enc["input_ids"], enc["attention_mask"])
            rewards = torch.tensor(
                [reward_of(t, enc["item"], tok, args.es_reward_weight) for t in comps],
                device=args.device, dtype=torch.float32)
            r_std = rewards.std(unbiased=False)
            sat = bool(r_std.item() < args.sat_std)
            adv = (rewards - rewards.mean()) / (r_std + 1e-8)
            buf.append({"prompt_len": enc["input_ids"].shape[1], "full_ids": full_ids,
                        "full_attention_mask": full_mask, "completion_mask": comp_mask,
                        "old_logprobs": old_lp, "advantages": adv, "rewards": rewards,
                        "sat": sat})

        step_r = float(torch.cat([b["rewards"] for b in buf]).mean().item())
        recent.append(step_r)
        if (len(recent) >= args.early_stop_window
                and sum(recent) / len(recent) >= args.early_stop_threshold):
            stopped_at, reason = step, "early_stop"
            print(f"[step {step}] early stop rolling_r={sum(recent)/len(recent):.3f}",
                  flush=True)
            break

        contrib = [b for b in buf if not b["sat"]]
        n_sat = len(buf) - len(contrib)
        if not contrib:
            tw.writerow([step, f"{time.time()-t0:.1f}", f"{step_r:.4f}", n_sat,
                         "", "", "", ""])
            train_log.flush()
            print(f"[step {step:3d}] mean_r={step_r:.3f} ALL SATURATED", flush=True)
            continue

        last = None
        gnorm = 0.0
        for ep in range(args.ppo_epochs):
            order = list(range(len(contrib)))
            random.Random(step * 1000 + ep).shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            for j in order:
                b = contrib[j]
                plp, klt = policy_forward_with_kl(
                    model, b["full_ids"], b["full_attention_mask"],
                    b["prompt_len"], b["completion_mask"])
                loss, diag = grpo_loss(plp, b["old_logprobs"], b["advantages"], klt,
                                       b["completion_mask"], args.clip_eps, args.kl_beta)
                (loss / len(contrib)).backward()
                last = diag
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip).item()
            optimizer.step()

        tw.writerow([step, f"{time.time()-t0:.1f}", f"{step_r:.4f}", n_sat,
                     f"{last['pg_loss']:.5f}", f"{last['kl_loss']:.5f}",
                     f"{last['clip_frac']:.3f}", f"{gnorm:.3f}"])
        train_log.flush()
        print(f"[step {step:3d}] mean_r={step_r:.3f} (sat={n_sat}/{len(buf)}) "
              f"pg={last['pg_loss']:+.4f} kl={last['kl_loss']:.4f} "
              f"clip={last['clip_frac']:.2f} grad={gnorm:.3f}", flush=True)

    train_log.close()
    print(f"\nStopped at step {stopped_at} ({reason}). Post-eval (n={args.n_eval})...",
          flush=True)

    gk = dict(max_new_tokens=128, temperature=1.0, top_p=0.9, eval_batch=args.eval_batch)
    post_qf = evaluate_set(model, tok, enc_qf, args.n_eval, label="post Q_F", **gk)
    post_qh = evaluate_set(model, tok, enc_qh, args.n_eval, label="post Q_held", **gk)
    write_csv(os.path.join(out_dir, "task5_post_qf.csv"), post_qf)
    write_csv(os.path.join(out_dir, "task5_post_qheld.csv"), post_qh)

    pre_qf = read_csv(args.pre_qf)
    pre_qh = read_csv(args.pre_qheld)

    def delta(label, pre, post):
        a, b = aggregate(pre), aggregate(post)
        print(f"\n  {label}: n={b['n_questions']}")
        for k, name in [("mean_p_hat", "p_hat"), ("mean_m_bin", "M_bin(mean)"),
                        ("median_m_bin", "M_bin(med)"), ("mean_m_mu", "M_mu"),
                        ("mean_es", "ES(mean)"), ("median_es", "ES(med)"),
                        ("frac_greedy_leak", "greedy")]:
            print(f"    {name:<12} {a[k]:.3f} -> {b[k]:.3f}  ({b[k]-a[k]:+.3f})")
        gap_pre = a["mean_m_bin"] - a["frac_greedy_leak"]
        gap_post = b["mean_m_bin"] - b["frac_greedy_leak"]
        print(f"    greedy-vs-prob gap (M_bin - greedy): {gap_pre:.3f} -> {gap_post:.3f}"
              f"  ({gap_post-gap_pre:+.3f})")

    print("\n" + "=" * 70 + "\nTASK 5 PRE->POST DELTAS\n" + "=" * 70)
    print_agg("PRE  Q_F", aggregate(pre_qf)); print_agg("POST Q_F", aggregate(post_qf))
    print_agg("PRE  Q_held", aggregate(pre_qh)); print_agg("POST Q_held", aggregate(post_qh))
    delta("Q_F (trained)", pre_qf, post_qf)
    delta("Q_held (generalization)", pre_qh, post_qh)

    adapter_dir = os.path.join(out_dir, "task5_adapter")
    model.save_pretrained(adapter_dir)
    print(f"\nSaved adapter to {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
