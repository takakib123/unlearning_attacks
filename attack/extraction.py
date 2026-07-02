"""
extraction.py
=============
Extraction Strength (ES) — a memorization metric that is gaming-resistant because
it requires the model to regenerate the EXACT ground-truth tokens (paraphrase /
hallucination score ~0).

    ES(x, y; theta) = 1 - (1/|y|) * min_k { k : greedy(f, [x, y_<k]) == y_>=k }

i.e. the smallest gold-answer prefix length k that lets the model greedily complete
the rest of y exactly; ES = 1 - k/|y| (higher = more memorized).

Exact single-forward computation (used for eval)
------------------------------------------------
Greedy decoding from [x, y_<k] reproduces the suffix y_>=k EXACTLY iff every gold
token from position k onward is the argmax under teacher forcing (each correct
argmax feeds the true next token, which then equals what greedy would emit). So the
smallest valid k is (index of the LAST argmax-wrong gold token) + 1. One teacher-
forced forward pass over [x, y] gives every argmax; ES follows. No |y|-fold decoding.

Rollout-level extraction (used for the GRPO reward)
---------------------------------------------------
ES above depends only on (x, y, theta) — constant across a GRPO rollout group, so it
cancels in the advantage. For the reward we use a rollout-DEPENDENT analogue:
es_rollout(y_hat, y) = (longest contiguous gold-token run reproduced verbatim in the
sampled completion y_hat) / |y|. Verbatim, so template mimicry scores low.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import List

import torch


@torch.no_grad()
def extraction_strength(model, tokenizer, prompt_text: str, answer_text: str,
                        device: str) -> float:
    """Exact ES via one teacher-forced forward pass. prompt_text is the full
    templated prompt (e.g. '[INST] q [/INST]'); answer_text is the gold answer."""
    prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                           add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt_text + " " + answer_text, return_tensors="pt",
                         add_special_tokens=True)["input_ids"]
    plen = prompt_ids.shape[1]
    L = full_ids.shape[1]
    ylen = L - plen
    if ylen <= 0:
        return 0.0
    full_ids = full_ids.to(device)
    logits = model(input_ids=full_ids).logits[0]  # (L, V)
    # gold token at absolute position p (p in [plen, L-1]) predicted from logits[p-1]
    pred = logits[plen - 1:L - 1].argmax(dim=-1)     # (ylen,)
    gold = full_ids[0, plen:L]                        # (ylen,)
    correct = (pred == gold)
    wrong_idx = (~correct).nonzero(as_tuple=True)[0]
    k = 0 if wrong_idx.numel() == 0 else int(wrong_idx.max().item()) + 1
    return 1.0 - k / ylen


def _tok_ids(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def es_rollout(pred_text: str, answer_text: str, tokenizer) -> float:
    """Rollout-level extraction reward: longest contiguous gold-token run present
    verbatim in the sampled completion, normalized by |gold tokens|. In [0, 1]."""
    gold = _tok_ids(tokenizer, answer_text)
    if not gold:
        return 0.0
    pred = _tok_ids(tokenizer, pred_text)
    if not pred:
        return 0.0
    m = SequenceMatcher(None, pred, gold, autojunk=False).find_longest_match(
        0, len(pred), 0, len(gold))
    return m.size / len(gold)
