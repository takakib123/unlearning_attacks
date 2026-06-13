"""
grpo_prototype.py
==================
Task 1 from `attack_protocol_grpo_v0.md`: sanity-check the GRPO loop.

Single prompt, single training task. We attack a base Phi-1.5 model
(no unlearning involved) to make it say 'cat' more often. Reward is
binary: 1 if 'cat' / 'cats' appears as a whole word in the completion
(case-insensitive), else 0.

This isolates the GRPO machinery from the unlearning question. If
reward goes up, KL stays bounded, and gradients flow, the next step
(single-question HP attack, Task 2) just swaps the model, the reward
function, and the dataset — the loop itself is unchanged.

Design choices that mirror `attack_protocol_grpo_v0.md`:

  - LoRA on q_proj / v_proj of the HF-native Phi-1.5 attention layers.
    (Do NOT pass trust_remote_code=True. The custom Microsoft code uses
    a fused Wqkv module which would require different LoRA targets.)
  - Reference policy = the same model with the LoRA adapter disabled
    via `peft_model.disable_adapter()`. Saves one model copy on GPU.
  - Exact per-token KL from full softmax distributions (no
    single-sample estimator).
  - Group-normalized advantage A_i = (r_i - mean(r)) / (std(r) + eps).
  - PPO-style clipped surrogate with K=4 epochs per rollout buffer so
    the importance ratio actually drifts away from 1 and clipping is
    exercised.
  - Sampling at T=1.0, top-p=0.9. Logprobs / KL computed on the FULL
    softmax (the standard PPO-for-LM convention; the top-p truncation
    is treated as a sampling-time artifact, not part of the policy).

Run:
    python grpo_prototype.py
    python grpo_prototype.py --num_outer_steps 5   # smoke test

Outputs:
    grpo_prototype_log.csv   - per-step metrics
    stdout                   - human-readable summary + periodic eval

Requires: torch, transformers >= 4.37, peft.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, TaskType, get_peft_model


# =====================================================================
# Config
# =====================================================================

@dataclass
class Config:
    # Model
    model_name: str = "microsoft/phi-1_5"
    device: str = "cuda"
    dtype_name: str = "bfloat16"

    # Prompt and target
    prompt: str = "Question: What is the cutest small furry pet?\nAnswer:"
    target_regex: str = r"\bcats?\b"

    # LoRA
    lora_rank: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj")

    # Sampling
    group_size: int = 8                  # G completions per rollout
    max_new_tokens: int = 32
    sampling_temperature: float = 1.0
    sampling_top_p: float = 0.9

    # GRPO
    num_outer_steps: int = 100           # rollout collection rounds
    ppo_epochs: int = 4                  # K: gradient steps per buffer
    clip_eps: float = 0.2
    kl_beta: float = 1e-2

    # Optim
    learning_rate: float = 1e-4
    grad_clip_norm: float = 1.0

    # Logging
    eval_every: int = 10
    eval_samples: int = 64
    seed: int = 0
    log_path: str = "grpo_prototype_log.csv"


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser()
    for f in dataclasses.fields(cfg):
        if f.type in (str, int, float):
            parser.add_argument(f"--{f.name}", type=f.type, default=getattr(cfg, f.name))
    args = parser.parse_args()
    for f in dataclasses.fields(cfg):
        if hasattr(args, f.name):
            setattr(cfg, f.name, getattr(args, f.name))
    return cfg


# =====================================================================
# Model setup
# =====================================================================

def load_model_and_tokenizer(cfg: Config):
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[cfg.dtype_name]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
    )
    base_model.to(cfg.device)

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# =====================================================================
# Reward
# =====================================================================

def reward_fn(completion_text: str, pattern: re.Pattern) -> float:
    return 1.0 if pattern.search(completion_text) else 0.0


# =====================================================================
# Sampling and logprob extraction
# =====================================================================

def build_completion_mask(completion_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """
    Mask completion tokens: 1 up to and including the first EOS, 0 after.

    Example
    -------
    completion = [t1, t2, EOS, pad, pad]   ->   mask = [1, 1, 1, 0, 0]
    """
    is_eos = (completion_ids == eos_token_id).long()
    cum_eos = is_eos.cumsum(dim=-1)
    prev_cum_eos = torch.cat(
        [torch.zeros_like(cum_eos[:, :1]), cum_eos[:, :-1]],
        dim=1,
    )
    return (prev_cum_eos == 0).float()


@torch.no_grad()
def sample_rollouts(
    model,
    tokenizer,
    cfg: Config,
    prompt_ids: torch.Tensor,         # [1, prompt_len]
    prompt_attention_mask: torch.Tensor,  # [1, prompt_len]
):
    """
    Sample G completions from the current policy. Recompute the sampling-time
    logprobs of each generated token by a single forward pass over the full
    sequence — these are the 'old' logprobs for the PPO importance ratio.

    Returns
    -------
    full_ids:            [G, prompt_len + completion_len]
    full_attention_mask: [G, prompt_len + completion_len]  (1 for real tokens, 0 for pad)
    completion_mask:     [G, completion_len]               (1 up to first EOS inclusive, 0 after)
    old_logprobs:        [G, completion_len]               (logprob under sampling-time policy)
    completions_text:    list[str]  of length G            (decoded completions)
    """
    G = cfg.group_size
    prompt_len = prompt_ids.shape[1]

    # Expand prompt to G copies.
    prompt_ids_g = prompt_ids.expand(G, -1).contiguous()
    prompt_mask_g = prompt_attention_mask.expand(G, -1).contiguous()

    model.eval()
    gen_out = model.generate(
        input_ids=prompt_ids_g,
        attention_mask=prompt_mask_g,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.sampling_temperature,
        top_p=cfg.sampling_top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )
    full_ids = gen_out.sequences                       # [G, prompt_len + new_len]
    completion_ids = full_ids[:, prompt_len:]          # [G, new_len]

    completion_mask = build_completion_mask(completion_ids, tokenizer.eos_token_id)

    # full_attention_mask: prompt portion = original mask, completion portion = completion_mask.
    full_attention_mask = torch.cat([prompt_mask_g, completion_mask.long()], dim=1)

    # Recompute logprobs of the sampled tokens under the current (sampling-time) policy.
    # Standard PPO convention: use full softmax, not top-p-truncated, for the ratio.
    outputs = model(input_ids=full_ids, attention_mask=full_attention_mask)
    logits = outputs.logits                            # [G, total_len, V]
    # Logits at position i predict token at position i+1. The completion tokens
    # live at positions [prompt_len, total_len); the logits that predict them
    # live at positions [prompt_len - 1, total_len - 1).
    completion_logits = logits[:, prompt_len - 1 : -1, :]   # [G, completion_len, V]
    log_probs = F.log_softmax(completion_logits.float(), dim=-1)
    old_logprobs = log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
    old_logprobs = old_logprobs * completion_mask

    completions_text = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in completion_ids
    ]
    model.train()

    return full_ids, full_attention_mask, completion_mask, old_logprobs.detach(), completions_text


# =====================================================================
# Loss
# =====================================================================

def policy_forward_with_kl(
    model,
    full_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    prompt_len: int,
    completion_mask: torch.Tensor,
):
    """
    One forward pass for policy logprobs (gradient enabled) and one for
    reference logprobs (no_grad, LoRA disabled), both sharing the base weights.

    Returns
    -------
    policy_logprobs: [G, completion_len]
    kl_per_token:    [G, completion_len]   exact KL(policy || ref) per position
    """
    completion_ids = full_ids[:, prompt_len:]

    # Policy forward (gradient enabled, LoRA active).
    policy_logits = model(
        input_ids=full_ids,
        attention_mask=full_attention_mask,
    ).logits[:, prompt_len - 1 : -1, :].float()

    # Reference forward (no gradient, LoRA disabled).
    with torch.no_grad(), model.disable_adapter():
        ref_logits = model(
            input_ids=full_ids,
            attention_mask=full_attention_mask,
        ).logits[:, prompt_len - 1 : -1, :].float()

    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # Per-token logprob of the actually-sampled tokens under the current policy.
    policy_logprobs = policy_log_probs.gather(
        -1, completion_ids.unsqueeze(-1)
    ).squeeze(-1)
    policy_logprobs = policy_logprobs * completion_mask

    # Exact per-token KL: sum_v p(v) (log p(v) - log q(v)).
    policy_probs = policy_log_probs.exp()
    kl_per_token = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)
    kl_per_token = kl_per_token * completion_mask

    return policy_logprobs, kl_per_token


def grpo_loss(
    policy_logprobs: torch.Tensor,    # [G, L]
    old_logprobs: torch.Tensor,       # [G, L]
    advantages: torch.Tensor,         # [G]
    kl_per_token: torch.Tensor,       # [G, L]
    completion_mask: torch.Tensor,    # [G, L]
    clip_eps: float,
    kl_beta: float,
):
    """
    PPO clipped surrogate + per-token KL penalty.

    Advantage is broadcast over all generated tokens of a completion
    (no per-token critic). All token-level means are mask-weighted.
    """
    # Importance ratio per token.
    log_ratio = policy_logprobs - old_logprobs
    ratio = log_ratio.exp()

    A = advantages.unsqueeze(-1)                       # [G, 1]
    unclipped = ratio * A
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A
    pg_per_token = -torch.minimum(unclipped, clipped)

    # Mask-weighted means.
    mask = completion_mask
    n_tokens = mask.sum().clamp_min(1.0)

    pg_loss = (pg_per_token * mask).sum() / n_tokens
    kl_loss = (kl_per_token * mask).sum() / n_tokens

    total = pg_loss + kl_beta * kl_loss

    # Diagnostics.
    with torch.no_grad():
        approx_kl_from_ratio = (((ratio - 1.0) - log_ratio) * mask).sum() / n_tokens
        clip_frac = (((ratio - 1.0).abs() > clip_eps).float() * mask).sum() / n_tokens

    return total, {
        "pg_loss": pg_loss.detach().item(),
        "kl_loss": kl_loss.detach().item(),
        "approx_kl_ratio": approx_kl_from_ratio.detach().item(),
        "clip_frac": clip_frac.detach().item(),
    }


# =====================================================================
# Eval
# =====================================================================

@torch.no_grad()
def eval_keyword_rate(model, tokenizer, cfg, prompt_ids, prompt_attention_mask, pattern):
    """Sample `cfg.eval_samples` completions and report fraction matching the target regex."""
    model.eval()
    n = cfg.eval_samples
    batch = min(n, 32)
    hits = 0
    total = 0
    while total < n:
        b = min(batch, n - total)
        out = model.generate(
            input_ids=prompt_ids.expand(b, -1).contiguous(),
            attention_mask=prompt_attention_mask.expand(b, -1).contiguous(),
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.sampling_temperature,
            top_p=cfg.sampling_top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        completions = out[:, prompt_ids.shape[1]:]
        for ids in completions:
            text = tokenizer.decode(ids, skip_special_tokens=True)
            if pattern.search(text):
                hits += 1
            total += 1
    model.train()
    return hits / total


# =====================================================================
# Main loop
# =====================================================================

def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    print(f"Config:\n{dataclasses.asdict(cfg)}")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU. This will be slow.")
        cfg.device = "cpu"
        cfg.dtype_name = "float32"

    pattern = re.compile(cfg.target_regex, re.IGNORECASE)

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )

    prompt_enc = tokenizer(
        cfg.prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(cfg.device)
    prompt_ids = prompt_enc["input_ids"]                   # [1, prompt_len]
    prompt_attention_mask = prompt_enc["attention_mask"]   # [1, prompt_len]
    prompt_len = prompt_ids.shape[1]
    print(f"Prompt ({prompt_len} tokens): {cfg.prompt!r}")

    # Initial eval.
    init_rate = eval_keyword_rate(model, tokenizer, cfg, prompt_ids, prompt_attention_mask, pattern)
    print(f"Initial 'cat' rate (n={cfg.eval_samples}): {init_rate:.3f}")

    log_file = open(cfg.log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "outer_step", "ppo_epoch", "wall_s",
        "reward_mean", "reward_std", "advantage_std",
        "pg_loss", "kl_loss", "approx_kl_ratio", "clip_frac",
        "grad_norm", "eval_rate",
    ])

    t_start = time.time()
    eval_rate = init_rate

    for step in range(cfg.num_outer_steps):
        # --- Rollout ---
        full_ids, full_attention_mask, completion_mask, old_logprobs, completions_text = sample_rollouts(
            model, tokenizer, cfg, prompt_ids, prompt_attention_mask
        )

        # --- Reward and advantage ---
        rewards = torch.tensor(
            [reward_fn(t, pattern) for t in completions_text],
            device=cfg.device,
            dtype=torch.float32,
        )
        r_mean = rewards.mean()
        r_std = rewards.std(unbiased=False)
        advantages = (rewards - r_mean) / (r_std + 1e-8)

        # --- K PPO epochs over the same buffer ---
        for ppo_epoch in range(cfg.ppo_epochs):
            policy_logprobs, kl_per_token = policy_forward_with_kl(
                model, full_ids, full_attention_mask, prompt_len, completion_mask
            )
            loss, diag = grpo_loss(
                policy_logprobs=policy_logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                kl_per_token=kl_per_token,
                completion_mask=completion_mask,
                clip_eps=cfg.clip_eps,
                kl_beta=cfg.kl_beta,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.grad_clip_norm,
            )
            optimizer.step()

            log_writer.writerow([
                step, ppo_epoch, f"{time.time() - t_start:.1f}",
                f"{r_mean.item():.4f}", f"{r_std.item():.4f}", f"{advantages.std(unbiased=False).item():.4f}",
                f"{diag['pg_loss']:.5f}", f"{diag['kl_loss']:.5f}",
                f"{diag['approx_kl_ratio']:.5f}", f"{diag['clip_frac']:.3f}",
                f"{grad_norm.item():.4f}", f"{eval_rate:.4f}",
            ])
            log_file.flush()

        # --- Eval (probabilistic 'cat' rate) ---
        do_eval = (step + 1) % cfg.eval_every == 0 or step == 0
        if do_eval:
            eval_rate = eval_keyword_rate(
                model, tokenizer, cfg, prompt_ids, prompt_attention_mask, pattern
            )

        # --- Print summary ---
        print(
            f"[step {step:3d}] "
            f"reward={r_mean.item():.3f}±{r_std.item():.3f} "
            f"pg={diag['pg_loss']:+.4f} "
            f"kl={diag['kl_loss']:.4f} "
            f"clip_frac={diag['clip_frac']:.2f} "
            f"approx_kl={diag['approx_kl_ratio']:.4f} "
            f"grad={grad_norm.item():.3f}"
            + (f"  eval='cat'={eval_rate:.3f}" if do_eval else "")
        )

        # Show one example completion every eval step.
        if do_eval:
            best_i = int(rewards.argmax().item())
            print(f"           sample[{best_i}] reward={rewards[best_i].item():.0f}: "
                  f"{completions_text[best_i]!r}")

    log_file.close()
    print(f"\nDone. Log written to {cfg.log_path}")
    final_rate = eval_keyword_rate(model, tokenizer, cfg, prompt_ids, prompt_attention_mask, pattern)
    print(f"Final 'cat' rate (n={cfg.eval_samples}): {final_rate:.3f}  (started at {init_rate:.3f})")


if __name__ == "__main__":
    main()